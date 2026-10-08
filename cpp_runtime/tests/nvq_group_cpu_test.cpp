#include "nvq_group.h"
#include "packed_nint.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace {
void set_single_rows(bool enabled) {
#ifdef _WIN32
    _putenv_s("MFQ_CPU_NVQ_SINGLE_ROW",enabled ? "1" : "0");
#else
    setenv("MFQ_CPU_NVQ_SINGLE_ROW",enabled ? "1" : "0",1);
#endif
}
// Every compressed buffer and activation ends immediately before a guard
// page, including partial final code words and 1..7 activation tails.
class Guarded {
    void* allocation_=nullptr;
    std::size_t size_=0;
public:
    unsigned char* data=nullptr;
    explicit Guarded(std::size_t bytes) {
#ifdef _WIN32
        SYSTEM_INFO info; GetSystemInfo(&info);
        const auto page=static_cast<std::size_t>(info.dwPageSize);
#else
        const auto page=static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
#endif
        const auto usable=((bytes+page-1)/page+1)*page;
        size_=usable+page;
#ifdef _WIN32
        allocation_=VirtualAlloc(nullptr,size_,MEM_RESERVE|MEM_COMMIT,PAGE_READWRITE);
        DWORD old;
        if (!allocation_ || !VirtualProtect(static_cast<char*>(allocation_)+usable,page,PAGE_NOACCESS,&old))
            throw std::runtime_error("guard allocation failed");
#else
        allocation_=mmap(nullptr,size_,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
        if (allocation_==MAP_FAILED || mprotect(static_cast<char*>(allocation_)+usable,page,PROT_NONE))
            throw std::runtime_error("guard allocation failed");
#endif
        data=static_cast<unsigned char*>(allocation_)+usable-bytes;
    }
    ~Guarded() {
#ifdef _WIN32
        VirtualFree(allocation_,0,MEM_RELEASE);
#else
        munmap(allocation_,size_);
#endif
    }
    Guarded(const Guarded&)=delete;
    Guarded& operator=(const Guarded&)=delete;
};

void encode(std::vector<std::uint8_t>& storage,std::size_t bit,int bits,unsigned value) {
    for (int i=0; i<bits; ++i)
        if ((value>>i)&1u) storage[(bit+i)/8]|=1u<<((bit+i)%8);
}

int parity(unsigned value) {
    int result=0;
    for (int i=0; i<7; ++i) result^=(value>>i)&1u;
    return result;
}

void check(int format,int width,int batch,int sign_mode) {
    const auto kernel=mfq::cpu::nvq_group_dot_kernel(format);
    if (!kernel) throw std::runtime_error("missing supported NVQ group kernel");
    const bool d4=format==3 || format==10 || format==11 || format==12 || format==15;
    const bool delta_format=format==1 || format==8;
    const bool no_aux=format==7 || format==9;
    const int bits=format==1 ? 11 : format==7 ? 7 : (format==8 || format==12) ? 9 :
        format==9 ? 6 : (format==13 || format==15) ? 10 : format==14 ? 12 : 8;
    const int vector_size=d4 ? 4 : 8;
    const int vectors=(width+vector_size-1)/vector_size, signs=(width+7)/8, groups=(width+23)/24;
    constexpr int rows=3, states=8;
    const int aux_count=no_aux ? 0 : rows*(delta_format ? groups : signs);
    const int aux_bits=delta_format ? 1 : 7;
    std::vector<unsigned> codes(rows*vectors), masks(aux_count), selectors(rows*groups);
    std::vector<std::uint8_t> indices((codes.size()*bits+7)/8), aux((masks.size()*aux_bits+7)/8);
    std::mt19937 random(20261005+format*width);
    for (std::size_t i=0; i<codes.size(); ++i) {
        codes[i]=random()&((1u<<bits)-1u);
        encode(indices,i*bits,bits,codes[i]);
    }
    for (std::size_t i=0; i<masks.size(); ++i) {
        masks[i]=random()&((1u<<aux_bits)-1u);
        encode(aux,i*aux_bits,aux_bits,masks[i]);
    }
    for (auto& value: selectors) value=random()%states;
    std::vector<std::uint8_t> packed_states((selectors.size()*3+7)/8);
    for (std::size_t i=0; i<selectors.size(); ++i) encode(packed_states,i*3,3,selectors[i]);
    std::vector<std::vector<std::int8_t>> tables(states);
    const int table_bytes=format==7 ? 768 : (1<<bits)*vector_size*(format==8 ? 2 : 1);
    for (auto& table: tables) {
        table.resize(table_bytes);
        for (auto& value: table) value=static_cast<std::int8_t>(int(random()%(format==8 ? 7 : 11))-(format==8 ? 3 : 5));
    }
    Guarded index_guard(indices.size()), aux_guard(aux.size());
    std::memcpy(index_guard.data,indices.data(),indices.size());
    if (!aux.empty()) std::memcpy(aux_guard.data,aux.data(),aux.size());
    mfq::cpu::NvqDecodeView view{index_guard.data,no_aux ? nullptr : aux_guard.data,
        static_cast<std::int64_t>(indices.size()),static_cast<std::int64_t>(aux.size()),
        vectors,signs,groups,sign_mode};
    for (int state=0; state<states; ++state)
        view.banks[state]=format==7 ? tables[0].data()+state*32 : tables[state].data();
    Guarded state_guard(packed_states.size());
    std::memcpy(state_guard.data,packed_states.data(),packed_states.size());
    const float anchors[rows]={.091738f,.028931f,-.007231f};
    view.states=state_guard.data; view.state_bytes=static_cast<std::int64_t>(packed_states.size());
    view.anchors=anchors; view.width=width; view.state_bits=3;
    for (int state=0; state<states; ++state) view.multipliers[state]=(state+1)/64.0f;
    const int stride=width+3;
    Guarded x_guard(static_cast<std::size_t>((batch-1)*stride+width)*sizeof(float));
    auto* x=reinterpret_cast<float*>(x_guard.data);
    for (int sample=0; sample<batch; ++sample)
        for (int column=0; column<width; ++column)
            x[sample*stride+column]=float(int(random()%65)-32)/16;
    std::vector<float> matrix_output(batch*(rows+2),-13);
    set_single_rows(false);
    const auto matrix=mfq::cpu::nvq_rows_dot_kernel(format);
    if (!matrix) throw std::runtime_error("missing supported NVQ matrix kernel");
    matrix(view,x,batch,stride,matrix_output.data(),rows+2,0,rows);
    set_single_rows(true);
    const auto single=mfq::cpu::nvq_rows_dot_kernel(format);
    if(!single || (format==7 ? single!=matrix : single==matrix))
        throw std::runtime_error("single-row NVQ test selected the wrong implementation or fallback");
    std::vector<float> actual_single(batch*(rows+2),-13);
    single(view,x,batch,stride,actual_single.data(),rows+2,0,rows);
    if(std::memcmp(actual_single.data(),matrix_output.data(),matrix_output.size()*sizeof(float)))
        throw std::runtime_error("single-row NVQ changed original FP32 output bits,format="+std::to_string(format)+" width="+std::to_string(width));
    std::fill(actual_single.begin(),actual_single.end(),-13);
    single(view,x,batch,stride,actual_single.data(),rows+2,1,2);
    for(int sample=0;sample<batch;++sample)for(int row=0;row<rows+2;++row) {
        const float expected=row==1 ? matrix_output[sample*(rows+2)+row] : -13;
        if(std::memcmp(&actual_single[sample*(rows+2)+row],&expected,sizeof(float)))
            throw std::runtime_error("single-row NVQ changed selected row range or output padding");
    }
    set_single_rows(false);
    for (int row=0; row<rows; ++row) {
        std::vector<float> actual(batch+2,13.0f);
        std::vector<double> reference(batch,13.0),magnitude(batch);
        for (int group=0; group<groups; ++group) {
            const auto state=selectors[row*groups+group];
            const int start=group*24, valid=std::min(24,width-start);
            const float scale=anchors[row]*((state+1)/64.0f);
            kernel(view,row,group,state,x+start,batch,stride,valid,scale,actual.data());
            // Oracle uses the original integer codes and signs supplied to
            // the encoder, with FP64 sums. It never reads packed storage.
            for (int column=start; column<start+valid; ++column) {
                const int vector=column/vector_size, lane=column%vector_size;
                const auto code=codes[row*vectors+vector];
                const int delta=delta_format && masks[row*groups+group] ? -1 : 1;
                int value;
                if (format==7) {
                    const int offset=column%8<4 ? state*32+(code&7u)*4+column%4 :
                        256+state*64+(code>>3)*4+column%4;
                    value=tables[0][offset];
                } else {
                    int offset=int(code)*vector_size+lane;
                    if (format==8 && delta<0) offset+=512*8;
                    value=tables[state][offset];
                }
                if (format==1) value=static_cast<std::int8_t>(8*value+delta);
                else if (format==8) value=static_cast<std::int8_t>(32*value+5*delta);
                else if (!no_aux) {
                    const auto mask7=masks[row*signs+column/8];
                    const int last=parity(mask7)^((format==2 && sign_mode) ? ((code>>7)&1u) : 0u);
                    const auto mask8=mask7|(last<<7);
                    if ((mask8>>(column%8))&1u) value=static_cast<std::int8_t>(-value);
                }
                const float weight=scale*value;
                for (int sample=0; sample<batch; ++sample) {
                    const double product=double(weight)*x[sample*stride+column];
                    reference[sample]+=product;
                    magnitude[sample]+=std::abs(product);
                }
            }
        }
        for (int sample=0; sample<batch; ++sample)
            if (std::abs(double(actual[sample])-reference[sample])>1e-5+2e-6*magnitude[sample])
                throw std::runtime_error("NVQ FP64 group oracle mismatch, format="+std::to_string(format));
        for (int sample=0; sample<batch; ++sample) {
            if (std::abs(double(matrix_output[sample*(rows+2)+row])-(reference[sample]-13))>1e-5+2e-6*magnitude[sample])
                throw std::runtime_error("NVQ FP64 matrix oracle mismatch, format="+std::to_string(format));
            if (matrix_output[sample*(rows+2)+rows]!=-13 || matrix_output[sample*(rows+2)+rows+1]!=-13)
                throw std::runtime_error("NVQ output padding overwritten");
        }
        if (actual[batch]!=13 || actual[batch+1]!=13)
            throw std::runtime_error("NVQ accumulator padding overwritten");
        kernel(view,row,0,0,nullptr,0,0,0,0,nullptr);
    }
}
} // namespace

int main() try {
    if (!mfq::cpu::packed_nint_has_avx2()) {
        std::cout << "NVQ AVX2 unavailable; portable decoder uses canonical adapter tests\n";
        return 0;
    }
    int cases=0;
    for (int format: {1,2,3,5,7,8,9,10,11,12,13,14,15})
        for (int width: {1,7,8,9,17,23,24,25,31,49,640,641,2560,2561})
            for (int batch: {1,7})
                for (int sign_mode: {0,1}) { check(format,width,batch,sign_mode); ++cases; }
    if (mfq::cpu::nvq_group_dot_kernel(4)) throw std::runtime_error("unsupported kernel format accepted");
    std::cout << "NVQ guarded FP64 group cases passed=" << cases << " single-row original FP32 bits/ranges/fallback exact=" << cases << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
