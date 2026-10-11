#include "mfq_cuda_quant_ops.h"
#include "mfq_cuda_nint_route_hint.h"
#include "mfq_cuda_moe_ops.h"
#include "nint_row_metadata.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace tb=mfq_tensor_backend;
constexpr int rows=17,local_experts=2,routes=10,experts=8;
struct Pool {
    int gs,ng,stride,input_rows;
    std::vector<uint8_t> bits,sub,minima;
    std::vector<int64_t> offsets;
    std::vector<float> scale,minimum,xscale;
    std::vector<int8_t> input;
    tb::Tensor q,qb,qo,s,mn,ns,nm,qx,xs,metadata;
    Pool(int group_size,int width,int inputs,int offset,int padding,int seed,int fixed_bits)
        :gs(group_size),ng((width+gs-1)/gs),input_rows(inputs) {
        const int total=rows*local_experts,k=ng*gs;
        bits.resize(total);offsets.resize(total);sub.resize(total*ng);minima.resize(total*ng);
        scale.resize(total);minimum.resize(total);
        uint64_t end=3;
        for(int row=0;row<rows;++row)end+=uint64_t(k)*(fixed_bits?fixed_bits:1+(row*3+seed)%8);
        stride=int((end+7)/8)+padding;
        std::vector<uint8_t> packed(offset+stride*local_experts);
        for(int expert=0;expert<local_experts;++expert) {
            uint64_t bit=3;
            for(int row=0;row<rows;++row) {
                const int neuron=expert*rows+row;
                bits[neuron]=fixed_bits?fixed_bits:1+(row*3+seed)%8;offsets[neuron]=bit;
                scale[neuron]=float(neuron%5+1)*.000123f;
                minimum[neuron]=float(neuron%7-3)*.000743f;
                for(int g=0;g<ng;++g){sub[neuron*ng+g]=(neuron*11+g*7)%127;minima[neuron*ng+g]=(neuron*3+g*17)%127;}
                for(int col=0;col<k;++col) {
                    const unsigned value=(neuron*17+col*23+col/3)&((1u<<bits[neuron])-1);
                    for(int b=0;b<bits[neuron];++b)if((value>>b)&1u)
                        packed[offset+expert*stride+(bit+uint64_t(col)*bits[neuron]+b)/8]|=
                            uint8_t(1u<<((bit+uint64_t(col)*bits[neuron]+b)&7));
                }
                bit+=uint64_t(k)*bits[neuron];
            }
        }
        input.resize(inputs*k);xscale.resize(inputs*ng);
        for(int i=0;i<inputs*k;++i)input[i]=i%k>=width ? 0 : int8_t((i*37+seed*11)%255-127);
        for(int i=0;i<inputs*ng;++i)xscale[i]=(i%7==0?0.f:float(i%5+1)*.000731f);
        q=tb::tensor(packed).to(tb::kCUDA).narrow(0,offset,stride*local_experts);
        qb=tb::tensor(bits).to(tb::kCUDA);qo=tb::tensor(offsets).to(tb::kCUDA);
        s=tb::tensor(sub).to(tb::kCUDA);mn=tb::tensor(minima).to(tb::kCUDA);
        ns=tb::tensor(scale).to(tb::kCUDA);nm=tb::tensor(minimum).to(tb::kCUDA);
        qx=tb::tensor(input).to(tb::kCUDA);xs=tb::tensor(xscale).to(tb::kCUDA);
        metadata=make_nint_row_metadata(tb::tensor(offsets),tb::tensor(scale),
            tb::tensor(minimum),qo.device());
    }
    void expected(int expert,int source,int row,double& value,double& magnitude)const {
        const int neuron=expert*rows+row,k=ng*gs;
        value=magnitude=0;
        for(int g=0;g<ng;++g) {
            int dot=0,sum=0;
            for(int i=0;i<gs;++i) {
                const int col=g*gs+i,w=(neuron*17+col*23+col/3)&((1u<<bits[neuron])-1);
                const int x=input[source*k+col];sum+=x;dot+=x*w;
            }
            const double term=double(xscale[source*ng+g])*(double(scale[neuron])*sub[neuron*ng+g]*dot-
                double(minimum[neuron])*minima[neuron*ng+g]*sum);
            value+=term;magnitude+=std::abs(term);
        }
    }
};
static void check_case(int tokens,int width,bool down,int offset,int padding,int fixed_bits,int& count) {
    std::vector<Pool> pools;
    for(int gs:{24,28,5})pools.emplace_back(gs,width,down?tokens*routes:tokens,offset,padding,int(pools.size()),fixed_bits);
    Pool replacement(24,width,down?tokens*routes:tokens,offset,padding,19,fixed_bits?0:5);
    const bool direct_test=std::getenv("MFQ_TEST_NINT_DIRECT")!=nullptr;
    const bool stage_test=std::getenv("MFQ_TEST_NINT_ROW_STAGE")!=nullptr;
    if(stage_test && !direct_test)throw std::runtime_error("row stage oracle requires direct dispatch");
    std::vector<Pool> refills;
    std::vector<tb::Tensor> original_packed;
    if(direct_test)for(int i=0;i<2;++i) {
        refills.emplace_back(pools[i].gs,width,pools[i].input_rows,offset,padding,i,1);
        original_packed.push_back(pools[i].q.clone());
    }
    std::vector<int64_t> pointers;std::vector<int32_t> parameters;
    for(auto& p:pools) {
        for(const auto* t:{&p.q,&p.qb,&p.qo,&p.s,&p.mn,&p.ns,&p.nm,&p.qx,&p.xs})pointers.push_back(reinterpret_cast<int64_t>(t->data_ptr()));
        parameters.insert(parameters.end(),{p.ng,p.gs,p.stride});
    }
    auto ptr=tb::tensor(pointers).reshape({3,9}).to(tb::kCUDA);
    auto params=tb::tensor(parameters).reshape({3,3}).to(tb::kCUDA);
    std::vector<int32_t> owner{0,1,2,0,1,2,-1,-1},local{0,0,0,1,1,1,-1,-1},selected(tokens*routes);
    auto op=tb::tensor(owner).to(tb::kCUDA),ol=tb::tensor(local).to(tb::kCUDA);
    std::array<tb::Tensor,2> direct_maps{ol.clone(),ol.clone()};
    auto remaining=op.clone();
    auto ids=tb::zeros({tokens,routes},op.options());
    auto output=tb::zeros({tokens,routes,rows},ptr.options().dtype(tb::kFloat16));
    auto stage_input=tb::zeros({1,width},output.options());
    auto run=[&] {
        if(cudaMemsetAsync(output.data_ptr(),0,output.numel()*output.element_size(),mfq_current_cuda_stream())!=cudaSuccess)
            throw std::runtime_error("output initialization failed");
        const char* hint_bits=std::getenv("MFQ_TEST_NINT_HINT_BITS");
        const char* hint_gs=std::getenv("MFQ_TEST_NINT_HINT_GS");
        if(direct_test) {
            for(int i=0;i<2;++i) {
                if(stage_test) {
                    if(cudaMemsetAsync(pools[i].metadata.data_ptr(),0,pools[i].metadata.nbytes(),mfq_current_cuda_stream())!=cudaSuccess)
                        throw std::runtime_error("row workspace initialization failed");
                    moe_quantize_shared_nint_rows_cuda(stage_input,{},0,false,pools[i].qo,pools[i].ns,pools[i].nm,
                        pools[i].metadata,direct_maps[i],ids,rows,local_experts);
                }
                NintSingleRouteWeight view{};
                std::copy_n(pointers.data()+i*9,9,view.pointers);
                std::copy_n(parameters.data()+i*3,3,view.geometry);
                view.local_experts=local_experts;
                view.row_metadata=reinterpret_cast<int64_t>(pools[i].metadata.data_ptr());
                if(!nint_try_single_route_cuda(view,direct_maps[i],ids,output,down,hint_bits?std::atoi(hint_bits):5,stage_test))
                    throw std::runtime_error("direct NINT oracle was not dispatched");
            }
            nint_moe_grouped_matmul_hetero_cuda(ptr,params,remaining,ol,ids,output,width,down);
        }
        else if(hint_bits && hint_gs)nint_moe_grouped_matmul_hinted_cuda(ptr,params,op,ol,ids,output,width,down,
            std::atoi(hint_gs),std::atoi(hint_bits));
        else nint_moe_grouped_matmul_hetero_cuda(ptr,params,op,ol,ids,output,width,down);
    };
    MfqCudaGraph graph;mfq_prepare_cuda_graph_memory(graph);run();
    if(cudaStreamSynchronize(mfq_current_cuda_stream())!=cudaSuccess)throw std::runtime_error("warmup failed");
    graph.capture_begin();run();graph.capture_end();
    for(int step=0;step<3;++step) {
        for(int i=0;i<tokens*routes;++i)selected[i]=(i*5+step*3)%11-1;
        owner[4]=step==1 ? -1 : 1;
        ids.copy_(tb::tensor(selected).reshape({tokens,routes}).to(tb::kCUDA));
        op.copy_(tb::tensor(owner).to(tb::kCUDA));
        // A graph may outlive an expert-cache refill. Replace the first
        // pool's row widths, packed bytes, geometry and activation pointers
        // together while retaining the captured host hint.
        auto replay_pointers=pointers;
        auto replay_parameters=parameters;
        if(step==1 && !direct_test) {
            int index=0;
            for(const auto* tensor:{&replacement.q,&replacement.qb,&replacement.qo,&replacement.s,&replacement.mn,
                    &replacement.ns,&replacement.nm,&replacement.qx,&replacement.xs})
                replay_pointers[index++]=reinterpret_cast<int64_t>(tensor->data_ptr());
            replay_parameters[0]=replacement.ng;replay_parameters[1]=replacement.gs;replay_parameters[2]=replacement.stride;
        }
        ptr.copy_(tb::tensor(replay_pointers).reshape({3,9}).to(tb::kCUDA));
        params.copy_(tb::tensor(replay_parameters).reshape({3,3}).to(tb::kCUDA));
        if(direct_test) {
            auto remaining_owner=owner;
            for(auto& value:remaining_owner)if(value!=2)value=-1;
            remaining.copy_(tb::tensor(remaining_owner).to(tb::kCUDA));
            for(int i=0;i<2;++i) {
                auto selected_local=local;
                for(int expert=0;expert<experts;++expert)if(owner[expert]!=i)selected_local[expert]=-1;
                direct_maps[i].copy_(tb::tensor(selected_local).to(tb::kCUDA));
                if(step==1) {
                    for(int expert=0;expert<local_experts;++expert)
                        pools[i].q.narrow(0,expert*pools[i].stride,refills[i].stride).copy_(
                            refills[i].q.narrow(0,expert*refills[i].stride,refills[i].stride));
                    pools[i].qb.copy_(refills[i].qb);pools[i].qo.copy_(refills[i].qo);
                    pools[i].metadata.copy_(refills[i].metadata);
                } else {
                    pools[i].q.copy_(original_packed[i]);
                    pools[i].qb.copy_(tb::tensor(pools[i].bits).to(tb::kCUDA));
                    pools[i].qo.copy_(tb::tensor(pools[i].offsets).to(tb::kCUDA));
                    pools[i].metadata.copy_(make_nint_row_metadata(tb::tensor(pools[i].offsets),
                        tb::tensor(pools[i].scale),tb::tensor(pools[i].minimum),pools[i].metadata.device()));
                }
            }
        }
        graph.replay();auto actual=output.to(tb::kCPU,tb::kFloat32);
        for(int pair=0;pair<tokens*routes;++pair)for(int row=0;row<rows;++row) {
            const int expert=selected[pair];double expected=0,magnitude=0;
            if(expert>=0 && expert<experts && owner[expert]>=0) {
                const auto& selected_pool=step==1 && direct_test && owner[expert]<2?refills[owner[expert]]:
                    (step==1 && !direct_test && owner[expert]==0?replacement:pools[owner[expert]]);
                selected_pool.expected(local[expert],down?pair:pair/routes,row,expected,magnitude);
            }
            const double value=actual.data_ptr<float>()[pair*rows+row];
            if(!std::isfinite(value) || std::abs(value-expected)>std::abs(expected)*.0005+magnitude*2e-6+6e-8)
                throw std::runtime_error("routed FP64 mismatch at pair="+std::to_string(pair)+" row="+std::to_string(row));
        }
        ++count;
    }
}
int main()try {
    int devices=0;const auto status=cudaGetDeviceCount(&devices);
    if(status==cudaErrorNoDevice || status==cudaErrorInsufficientDriver || (status==cudaSuccess && !devices))return 77;
    if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
    auto context=mfq::cuda::default_context(0);auto stream=mfq_get_stream_from_pool(false);MfqCudaStreamGuard guard(stream);
    int cases=0;
    for(int tokens:{1,3,8})for(int width:{1,25,640,2560})for(bool down:{false,true})
        for(int offset:{0,1})for(int padding:{0,8})for(int bits:{0,4,5,6})check_case(tokens,width,down,offset,padding,bits,cases);
    std::cout<<"NINT routed FP64 oracle: "<<cases<<" cases, mixed widths/groups, tails, offsets, mutable pool descriptors, masked/changing routes PASS\n";
    return 0;
}catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
