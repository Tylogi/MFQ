#include "packed_bench_utils.h"
#include "nvq_group.h"
#include "packed_nint.h"
#include "mfq/cpu_expert_pool.h"

#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>

namespace {
using Clock=std::chrono::steady_clock;
using mfq::cpu::NvqDecodeView;
using mfq::cpu::NvqRowsDot;
void select(bool enabled) {
#ifdef _WIN32
    _putenv_s("MFQ_CPU_NVQ_SINGLE_ROW",enabled ? "1" : "0");
#else
    setenv("MFQ_CPU_NVQ_SINGLE_ROW",enabled ? "1" : "0",1);
#endif
}
struct Weight {
    NvqDecodeView view{};
    std::vector<std::uint8_t> indices,aux,states;
    std::array<std::vector<std::int8_t>,16> banks;
    std::vector<float> anchors,reference;
    std::size_t bytes=0;
    Weight(int format,int rows,int width,std::uint32_t seed) {
        const bool d4=format==3 || format==10 || format==11 || format==12 || format==15;
        const bool delta=format==1 || format==8,no_aux=format==7 || format==9;
        const int bits=format==1 ? 11 : format==7 ? 7 : (format==8 || format==12) ? 9 :
            format==9 ? 6 : (format==13 || format==15) ? 10 : format==14 ? 12 : 8;
        const int vector_size=d4 ? 4 : 8;
        view.nvec=(width+vector_size-1)/vector_size;view.nsign=(width+7)/8;
        view.groups=(width+23)/24;view.width=width;view.state_bits=format==9 ? 2 : format==7 ? 3 : 4;
        indices.resize((std::size_t(rows)*view.nvec*bits+7)/8);
        aux.resize(no_aux ? 0 : (std::size_t(rows)*(delta ? view.groups : view.nsign)*(delta ? 1 : 7)+7)/8);
        states.resize((std::size_t(rows)*view.groups*view.state_bits+7)/8);
        mfq::bench::fill(indices,seed);mfq::bench::fill(aux,seed);mfq::bench::fill(states,seed);
        const int table_bytes=format==7 ? 768 : (1<<bits)*vector_size*(format==8 ? 2 : 1);
        const bool jsc=format==5 || format>=10;
        const int state_count=1<<view.state_bits,bank_count=jsc ? 2 : state_count;
        for(int state=0;state<bank_count;++state) {
            auto& bank=banks[state];bank.resize(table_bytes);
            for(auto& value:bank)value=static_cast<std::int8_t>(int(mfq::bench::random_word(seed)%11)-5);
        }
        for(int state=0;state<state_count;++state) {
            view.banks[state]=format==7 ? banks[0].data()+state*32 : banks[state%bank_count].data();
            view.multipliers[state]=float(state+1)/64;
        }
        anchors.resize(rows);reference.resize(rows);
        for(int row=0;row<rows;++row)anchors[row]=.023917f*float(row%11-5);
        view.indices=indices.data();view.aux=no_aux ? nullptr : aux.data();
        view.index_bytes=static_cast<std::int64_t>(indices.size());view.aux_bytes=static_cast<std::int64_t>(aux.size());
        view.states=states.data();view.state_bytes=static_cast<std::int64_t>(states.size());view.anchors=anchors.data();
        bytes=indices.size()+aux.size()+states.size()+anchors.size()*sizeof(float);
        for(const auto& bank:banks)bytes+=bank.size();
    }
};
void check_bits(NvqRowsDot kernel,Weight& w,const std::vector<float>& input,std::vector<float>& output,int rows,bool make_reference) {
    kernel(w.view,input.data(),1,w.view.width,output.data(),rows,0,rows);
    if(make_reference)w.reference=output;
    else if(std::memcmp(output.data(),w.reference.data(),std::size_t(rows)*sizeof(float)))
        throw std::runtime_error("CPU NVQ benchmark original FP32 bits mismatch");
}
void run_case(int format,int rows,int width,const mfq::bench::Settings& settings) {
    select(false);const auto off=mfq::cpu::nvq_rows_dot_kernel(format);
    select(true);const auto on=mfq::cpu::nvq_rows_dot_kernel(format);
    if(!off || !on || (format==7 ? off!=on : off==on))throw std::runtime_error("CPU NVQ benchmark path not selected");
    std::vector<float> input(width),output(rows);
    for(int i=0;i<width;++i)input[i]=std::sin(float(i+1)*.173f)*.31f;
    std::vector<std::unique_ptr<Weight>> weights;
    weights.emplace_back(std::make_unique<Weight>(format,rows,width,20261006u+format));
    const auto copies=std::max<std::size_t>(2,(settings.working_bytes+weights[0]->bytes-1)/weights[0]->bytes);
    for(std::size_t i=1;i<copies;++i)weights.emplace_back(std::make_unique<Weight>(format,rows,width,20261006u+format+std::uint32_t(i)*97));
    for(auto& weight:weights) {check_bits(off,*weight,input,output,rows,true);check_bits(on,*weight,input,output,rows,false);}
    mfq::cpu::ExpertPool pool(settings.threads-1);
    std::vector<int> thread_counts{1};
    if(settings.threads>1)thread_counts.push_back(settings.threads);
    for(const bool rotating:{false,true})for(const int threads:thread_counts) {
        const auto count=rotating ? weights.size() : std::size_t(1);
        auto measure=[&](NvqRowsDot kernel,std::size_t repeats) {
            const auto start=Clock::now();
            for(std::size_t repeat=0;repeat<repeats;++repeat)for(std::size_t index=0;index<count;++index) {
                auto& w=*weights[index];
                if(threads==1)kernel(w.view,input.data(),1,width,output.data(),rows,0,rows);
                else pool.rows(rows,[&](std::int64_t begin,std::int64_t end) {
                    kernel(w.view,input.data(),1,width,output.data(),rows,begin,end);
                });
            }
            const auto elapsed=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
            if(std::memcmp(output.data(),weights[count-1]->reference.data(),std::size_t(rows)*sizeof(float)))
                throw std::runtime_error("CPU NVQ timed serial/pool result changed original bits");
            return elapsed;
        };
        measure(off,1);measure(on,1);
        std::size_t repeats=1;
        while(std::min(measure(off,repeats),measure(on,repeats))<settings.target_ms) {
            if(repeats>std::numeric_limits<std::size_t>::max()/2)throw std::overflow_error("CPU benchmark repeats overflow");
            repeats*=2;
        }
        std::vector<double> a,b;
        for(int sample=0;sample<settings.samples;++sample) {
            double x,y;
            if(sample%2) {y=measure(on,repeats);x=measure(off,repeats);}
            else {x=measure(off,repeats);y=measure(on,repeats);}
            a.push_back(x*1000/(repeats*count));b.push_back(y*1000/(repeats*count));
        }
        for(auto& weight:weights)check_bits(on,*weight,input,output,rows,false);
        std::cout<<"{\"backend\":\"cpu\",\"operator\":\"nvq_gemv\",\"format\":"<<format
                 <<",\"rows\":"<<rows<<",\"width\":"<<width<<",\"M\":1,\"threads\":"<<threads
                 <<",\"timing_scope\":\""<<(threads==1 ? "kernel_wall" : "production_pinned_pool_wall")
                 <<"\",\"cache\":\""<<(rotating ? "rotating" : "hot")<<"\",\"working_sets\":"<<count
                 <<",\"weight_bytes\":"<<weights[0]->bytes<<",\"working_bytes\":"<<weights[0]->bytes*count
                 <<",\"repeats\":"<<repeats<<",\"samples\":"<<settings.samples
                 <<",\"selected_original_fallback\":"<<(off==on ? "true" : "false")<<",\"exact_original_bits\":true";
        mfq::bench::timing(a,b);std::cout<<"}\n"<<std::flush;
    }
    select(false);
}
} // namespace
int main(int argc,char** argv) try {
    const auto settings=mfq::bench::settings(argc,argv);
    if(!mfq::cpu::packed_nint_has_avx2())throw std::runtime_error("AVX2 unavailable");
    std::cout<<std::setprecision(9);
    const std::vector<int> formats=settings.all_formats ? std::vector<int>{1,2,3,5,7,8,9,10,11,12,13,14,15} : std::vector<int>{11,12,15};
    for(const int format:formats) {
        run_case(format,640,2560,settings);run_case(format,2560,640,settings);
    }
    return 0;
}catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
