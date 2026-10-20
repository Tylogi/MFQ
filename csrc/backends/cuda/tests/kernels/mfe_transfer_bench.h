#pragma once
#include "kernels/mfe/ffn_prepare.h"

static void benchmark_mfe_transfer(bool timings=true) {
    namespace tb=mfq_tensor_backend;
    auto context=default_context();const auto stream=mfq_current_cuda_stream();context->begin_graph_pool(stream);
    for(int tokens:{1,2,6,32}) {
        constexpr int experts=1536,routes=10;const int pairs=tokens*routes;
        constexpr int snapshot_fields=33,snapshot_words=4+snapshot_fields*4;
        const auto control_bytes=(experts*2+pairs*2+64)*sizeof(int32_t);
        const auto descriptor_words=uint64_t(pairs)*3*sizeof(MfePackedProjection)/sizeof(uint64_t);
        HostBuffer host(control_bytes+(snapshot_words+descriptor_words)*sizeof(uint64_t),true);
        auto* values=static_cast<int32_t*>(host.data());int32_t* mapped=nullptr;
        MFQ_CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&mapped),host.data(),0));
        for(int e=0;e<experts;++e){values[e]=e%3;values[experts+e]=e%3==2?e+7:-1;}
        std::vector<int32_t> routes_host(pairs);
        for(int i=0;i<pairs;++i) {
            const int expert=i%17==0?-1:i%23==0?experts+2:(i%11)*131;
            routes_host[i]=expert;values[experts*2+i*2]=expert>=0 && expert<experts?values[expert]:-1;
            values[experts*2+i*2+1]=expert>=0 && expert<experts?values[experts+expert]:-1;
        }
        const int flag_offset=experts*2+pairs*2;
        values[flag_offset]=values[flag_offset+16]=1;values[flag_offset+32]=0;
        auto* table=reinterpret_cast<uint64_t*>(static_cast<uint8_t*>(host.data())+control_bytes);
        auto* mapped_table=reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(mapped)+control_bytes);
        table[0]=snapshot_fields;table[1]=snapshot_fields*16;table[2]=table[3]=0;
        for(int word=4;word<snapshot_words;++word)table[word]=uint64_t(word)*78374791;
        auto* descriptors=table+snapshot_words;
        auto* mapped_descriptors=mapped_table+snapshot_words;
        for(uint64_t word=0;word<descriptor_words;++word)descriptors[word]=(word+1)*812733719;
        auto ids=tb::tensor(routes_host).to(tb::kCUDA);
        auto snapshot=tb::empty({snapshot_words},ids.options().dtype(tb::kInt64));
        auto device_descriptors=tb::empty({int64_t(descriptor_words+1)},snapshot.options());
        auto kinds=tb::empty({experts},ids.options()),indices=tb::empty_like(kinds),abort=tb::empty({1},ids.options());
        MfeFfnBatch batch;batch.tokens=tokens;batch.routes=routes;batch.experts=experts;
        batch.ids=ids.data_ptr<int32_t>();batch.kinds=kinds.data_ptr<int32_t>();batch.transfer_index=indices.data_ptr<int32_t>();
        batch.aborted=reinterpret_cast<uint32_t*>(abort.data_ptr());batch.resident_plan_overlap=true;
        batch.host_kinds=mapped;batch.host_transfer_index=mapped+experts;
        batch.plan_ready=reinterpret_cast<uint32_t*>(mapped+flag_offset);
        batch.transfer_ready=reinterpret_cast<uint32_t*>(mapped+flag_offset+16);
        batch.host_aborted=reinterpret_cast<uint32_t*>(mapped+flag_offset+32);
        for(bool cancelled:{false,true}) {
            values[flag_offset+32]=cancelled;
            tb::Tensor reference_kinds,reference_indices;
            for(int arm:{0,1,2,3,0}) {
                table[2]=arm==3?descriptor_words:0;
                const auto body=[&]{
                    if(arm==3)mfe_ffn_wait_transfer_packet_snapshot(batch,mapped+experts*2,mapped_table,
                        snapshot.data_ptr(),snapshot_fields,stream,mapped_descriptors,device_descriptors.data_ptr(),descriptor_words);
                    else if(arm==2)mfe_ffn_wait_transfer_packet_snapshot(batch,mapped+experts*2,mapped_table,
                        snapshot.data_ptr(),snapshot_fields,stream);
                    else if(arm)mfe_ffn_wait_transfer_packet(batch,mapped+experts*2,stream);
                    else mfe_ffn_wait_transfer(batch,stream);
                };
                kinds.fill_(6);indices.fill_(-99);abort.zero_();device_descriptors.fill_(-1);body();
                auto actual_kinds=kinds.cpu(),actual_indices=indices.cpu();
                check(abort.item<int32_t>()==int(cancelled),"compact transfer plan lost cancellation");
                if(arm>=2) {
                    auto copy=snapshot.cpu();auto* data=reinterpret_cast<uint64_t*>(copy.data_ptr());
                    check(data[0]==(cancelled?0:snapshot_fields) && data[2]==uint64_t(cancelled),"transfer snapshot lost cancellation");
                    if(!cancelled)for(int word=0;word<snapshot_words;++word)
                        if(word!=2)check(data[word]==table[word],"transfer wait changed a snapshot word");
                    table[0]=~uint64_t(0);body();
                    auto empty=snapshot.cpu();const auto* empty_words=reinterpret_cast<uint64_t*>(empty.data_ptr());
                    check(empty_words[0]==0 && empty_words[2]==uint64_t(cancelled?1:2) &&
                        empty_words[3]==uint64_t(!cancelled),"transfer snapshot lost empty-packet/cancellation marker");
                    table[0]=snapshot_fields;table[3]=0;
                }
                if(arm==3) {
                    auto actual=device_descriptors.cpu();const auto* data=reinterpret_cast<const uint64_t*>(actual.data_ptr());
                    for(uint64_t word=0;word<=descriptor_words;++word)
                        check(data[word]==(cancelled || word==descriptor_words?~uint64_t(0):descriptors[word]),
                            "mapped descriptor snapshot changed data, cancellation or padding");
                    table[2]=descriptor_words+1;device_descriptors.fill_(-1);body();
                    check(abort.item<int32_t>()==1,"mapped descriptor snapshot ignored its capacity");
                    auto untouched=device_descriptors.cpu();for(int64_t word=0;word<untouched.numel();++word)
                        check(untouched.data_ptr<int64_t>()[word]==-1,"overflowing descriptor packet wrote a partial image");
                    table[2]=descriptor_words;
                    Graph replay;replay.capture_begin_shared();body();replay.capture_end();
                    for(uint64_t length:{uint64_t(0),descriptor_words/2,descriptor_words}) {
                        table[2]=length;device_descriptors.fill_(-1);
                        for(uint64_t word=0;word<descriptor_words;++word)descriptors[word]^=uint64_t(length+17);
                        replay.replay();auto changed=device_descriptors.cpu();
                        const auto* current=reinterpret_cast<const uint64_t*>(changed.data_ptr());
                        for(uint64_t word=0;word<=descriptor_words;++word)
                            check(current[word]==(cancelled || word>=length?~uint64_t(0):descriptors[word]),
                                "descriptor graph replay kept stale words or copied beyond the active packet");
                    }
                }
                if(!reference_kinds.defined()){reference_kinds=actual_kinds;reference_indices=actual_indices;}
                else {
                    check(!std::memcmp(reference_kinds.data_ptr(),actual_kinds.data_ptr(),actual_kinds.nbytes()),"compact transfer plan changed kinds");
                    check(!std::memcmp(reference_indices.data_ptr(),actual_indices.data_ptr(),actual_indices.nbytes()),"compact transfer plan changed transfer indices");
                }
                if(!timings || cancelled)continue;
                Graph graph;graph.capture_begin_shared();for(int i=0;i<16;++i)body();graph.capture_end();
                for(int i=0;i<8;++i)graph.replay();
                cudaEvent_t begin,end;MFQ_CUDA_CHECK(cudaEventCreate(&begin));MFQ_CUDA_CHECK(cudaEventCreate(&end));
                MFQ_CUDA_CHECK(cudaEventRecord(begin,stream));for(int i=0;i<32;++i)graph.replay();
                MFQ_CUDA_CHECK(cudaEventRecord(end,stream));MFQ_CUDA_CHECK(cudaEventSynchronize(end));
                float ms=0;MFQ_CUDA_CHECK(cudaEventElapsedTime(&ms,begin,end));
                MFQ_CUDA_CHECK(cudaEventDestroy(begin));MFQ_CUDA_CHECK(cudaEventDestroy(end));
                std::cout<<"mfe_transfer tokens="<<tokens<<" compact="<<arm<<" us="<<ms*1000/(32*16)<<"\n";
            }
        }
    }
    // All-resident GPU classifications can avoid rereading the mapped route
    // packet, while retaining the CPU ownership acknowledgement and abort.
    {
        constexpr int experts=17,routes=10,words=4+4*8;
        HostBuffer host(1024,true);std::memset(host.data(),0,host.size());
        auto* cpu=static_cast<uint32_t*>(host.data());uint32_t* gpu=nullptr;
        MFQ_CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&gpu),cpu,0));
        auto* packet=reinterpret_cast<int32_t*>(cpu+64);auto* table=reinterpret_cast<uint64_t*>(cpu+96);
        std::vector<int32_t> input(routes);for(int i=0;i<routes;++i)input[i]=i%7;
        auto ids=tb::tensor(input).to(tb::kCUDA);
        auto kinds=tb::empty({experts},ids.options()),indices=tb::empty_like(kinds);
        auto abort=tb::zeros({1},ids.options()),snapshot=tb::empty({words},ids.options().dtype(tb::kInt64));
        MfeFfnBatch batch;batch.tokens=1;batch.routes=routes;batch.experts=experts;
        batch.ids=ids.data_ptr<int32_t>();batch.kinds=kinds.data_ptr<int32_t>();batch.transfer_index=indices.data_ptr<int32_t>();
        batch.aborted=reinterpret_cast<uint32_t*>(abort.data_ptr());batch.resident_plan_overlap=true;
        batch.transfer_ready=gpu;batch.host_aborted=gpu+16;cpu[0]=1;
        const auto select=[&](bool enabled) {
#ifdef _WIN32
            _putenv_s("MFQ_MFE_DEVICE_RESIDENT_PACKET",enabled?"1":"0");
#else
            setenv("MFQ_MFE_DEVICE_RESIDENT_PACKET",enabled?"1":"0",1);
#endif
        };
        for(bool enabled:{false,true}) {
            select(enabled);
            const auto body=[&]{mfe_ffn_wait_transfer_packet_snapshot(batch,reinterpret_cast<int32_t*>(gpu+64),
                gpu+96,snapshot.data_ptr(),8,stream);};
            Graph graph;graph.capture_begin_shared();body();graph.capture_end();
            for(bool resident:{true,false,true})for(bool cancelled:{false,true}) {
                cpu[16]=cancelled;table[0]=resident?~uint64_t(0):8;table[1]=resident?0:128;
                for(int word=4;word<words;++word)table[word]=uint64_t(word)*9231;
                for(int i=0;i<routes;++i){packet[i*2]=resident?1:2;packet[i*2+1]=resident?-1:input[i]+7;}
                kinds.fill_(resident?1:6);indices.fill_(-99);graph.replay();
                const auto actual=kinds.cpu(),slots=indices.cpu(),header=snapshot.cpu();
                check(abort.item<int32_t>()==int(cancelled),"resident packet lost cancellation");
                for(auto expert:input) {
                    check(actual.data_ptr<int32_t>()[expert]==(cancelled?-1:resident?1:6),"resident packet changed route classification");
                    check(slots.data_ptr<int32_t>()[expert]==(cancelled || resident?-1:expert+7),"resident packet changed transfer slot");
                }
                const auto* h=reinterpret_cast<const uint64_t*>(header.data_ptr());
                check(h[0]==uint64_t(cancelled || resident?0:8) && h[2]==uint64_t(cancelled?1:resident?2:0),
                    "resident/mixed packet graph replay kept a stale copy guard");
                if(!cancelled && !resident)for(int word=4;word<words;++word)
                    check(h[word]==table[word],"resident packet skipped mixed-plan fields");
            }
            cpu[16]=0;table[0]=~uint64_t(0);table[1]=0;kinds.fill_(1);
            for(int i=0;i<routes;++i){packet[i*2]=1;packet[i*2+1]=-1;}
            if(timings) {
                Graph timing;timing.capture_begin_shared();for(int i=0;i<16;++i)body();timing.capture_end();
                for(int i=0;i<8;++i)timing.replay();
                cudaEvent_t a,b;MFQ_CUDA_CHECK(cudaEventCreate(&a));MFQ_CUDA_CHECK(cudaEventCreate(&b));
                MFQ_CUDA_CHECK(cudaEventRecord(a,stream));for(int i=0;i<64;++i)timing.replay();
                MFQ_CUDA_CHECK(cudaEventRecord(b,stream));MFQ_CUDA_CHECK(cudaEventSynchronize(b));
                float ms=0;MFQ_CUDA_CHECK(cudaEventElapsedTime(&ms,a,b));
                cudaEventDestroy(a);cudaEventDestroy(b);
                std::cout<<"mfe_resident_packet enabled="<<enabled<<" us="<<ms*1000/(64*16)<<std::endl;
            }
        }
        // No mapped descriptors or route packet are consumed in the proven
        // resident case. Memcheck also verifies this with null addresses.
        kinds.fill_(1);mfe_ffn_wait_transfer_packet_snapshot(batch,nullptr,nullptr,snapshot.data_ptr(),8,stream);
        MFQ_CUDA_CHECK(cudaStreamSynchronize(stream));
        check(abort.item<int32_t>()==0,"resident packet rejected unused mapped data");
        select(false);
        std::cout<<"mfe_resident_packet hot/mixed replay and cancellation exact PASS"<<std::endl;
    }
    context->end_graph_pool(stream);
    std::cout<<"mfe_transfer packet duplicate/invalid routes, partial GU tags, descriptor snapshots, capacity, replay, cancellation exact PASS"<<std::endl;
}
