#include "quant/mfe_expert_store.h"
#include "mfq/host_parallel.h"
#include "quant/packed_nint_expert_store.h"
#include "model_source.h"
#include "../quant/nint_row_fixture.h"

#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace {
void require(bool condition,const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template<class F> void rejected(F f) {
    bool failed=false; try { f(); } catch (const std::exception&) { failed=true; }
    require(failed,"invalid expert range source accepted");
}
void pack(std::vector<std::uint8_t>& bytes,const std::vector<unsigned>& values,int bits) {
    const auto base=bytes.size(); bytes.resize(base+(values.size()*bits+7)/8,0);
    for (std::size_t index=0; index<values.size(); ++index)
        for (int bit=0; bit<bits; ++bit)
            bytes[base+(index*bits+bit)/8]|=((values[index]>>bit)&1u)<<((index*bits+bit)&7);
}
std::vector<std::uint8_t> nvq(int begin,int end) {
    constexpr int width=53,groups=3,vectors=7;
    const auto rows=end-begin;
    std::vector<std::uint8_t> blob{'N','Q','1','L',1,3,24,0};
    mfq::test::append<std::int32_t>(blob,0);
    mfq::test::append<std::int32_t>(blob,width);
    mfq::test::append<std::uint32_t>(blob,2);
    mfq::test::append<std::int64_t>(blob,rows);
    mfq::test::append<std::int64_t>(blob,width);
    mfq::test::append<std::uint32_t>(blob,rows);
    for (int row=begin; row<end; ++row) mfq::test::append<std::uint16_t>(blob,0x3c00+row);
    for (auto geometry: {std::pair<int,int>{groups,3},{vectors,11},{groups,1}}) {
        std::vector<unsigned> values;
        for (int row=begin; row<end; ++row)
            for (int column=0; column<geometry.first; ++column)
                values.push_back((row*53+column*17+7)&((1u<<geometry.second)-1));
        pack(blob,values,geometry.second);
    }
    return blob;
}
void check_group64_fields(int width) {
    constexpr int rows=34;
    const int groups=(width+23)/24,vectors=(width+7)/8;
    constexpr std::size_t anchor_offset=40+64+4096*8,record_offset=anchor_offset+rows*2;
    std::vector<std::uint8_t> blob{'N','V','Q','1',37,4,24,0};
    mfq::test::append<std::int32_t>(blob,0);
    mfq::test::append<std::int32_t>(blob,width);
    mfq::test::append<std::uint32_t>(blob,2);
    mfq::test::append<std::int64_t>(blob,rows);
    mfq::test::append<std::int64_t>(blob,width);
    mfq::test::append<std::uint32_t>(blob,rows);
    blob.resize(anchor_offset,0);blob[40]=2;blob[41]=1;blob[42]=16;blob[92]=1;
    for(int row=0;row<rows;++row)mfq::test::append<std::uint16_t>(blob,0x3c00+row);
    for(int row=0;row<rows;++row)for(int group=0;group<groups;++group) {
        std::uint64_t record=std::uint64_t((row*5+group*3)&15)<<60;
        for(int local=0;local<3;++local) {
            const auto vector=group*3+local;
            if(vector>=vectors)continue;
            const auto index=(row*53+vector*17+7)&4095,sign=(row*19+vector*11+3)&127;
            unsigned parity=0;for(int bit=0;bit<7;++bit)parity^=(sign>>bit)&1;
            const auto segment=std::uint64_t(index)|std::uint64_t(sign|(parity<<7))<<12;
            record|=segment<<(local*20);
        }
        mfq::test::append<std::uint64_t>(blob,record);
    }
    const auto read=[&](std::size_t offset,std::uint8_t* output,std::size_t count) {
        require(offset<=blob.size() && count<=blob.size()-offset,"group64 read exceeded source");
        std::memcpy(output,blob.data()+offset,count);
    };
    int batches=0;
    const mfq::NvqRows source(blob.size(),read,[&](const std::vector<mfq::ReadSpan>& spans) {
        require(spans.size()==2,"group64 fields were not read together");++batches;
        for(const auto& span:spans)read(span.offset,reinterpret_cast<std::uint8_t*>(span.destination),span.size);
    });
    require(source.group64(),"group64 row storage was not detected");
    for(int count=1;count<=rows;++count)
        require(source.byte_aligned_rows(count)==(((std::uint64_t(count)*groups*4 |
            std::uint64_t(count)*vectors*12 | std::uint64_t(count)*vectors*7)&7)==0),
            "group64 field concatenation ignored packed bit boundaries");
    rejected([&]{source.copy_rows_fields(0,1,{});});
    for(const auto range:{std::pair<int,int>{0,1},{1,18},{17,34},{0,34}}) {
        const auto fields=source.slice_rows_fields(range.first,range.second);
        require(bool(fields),"group64 fields were unavailable");
        std::vector<unsigned> states,indices,signs;
        for(int row=range.first;row<range.second;++row) {
            require(fields->neuron_scale[row-range.first]==0x3c00+row,"group64 anchor changed");
            for(int group=0;group<groups;++group)states.push_back((row*5+group*3)&15);
            for(int vector=0;vector<vectors;++vector) {
                indices.push_back((row*53+vector*17+7)&4095);
                signs.push_back((row*19+vector*11+3)&127);
            }
        }
        std::vector<std::uint8_t> a,b,c;pack(a,states,4);pack(b,indices,12);pack(c,signs,7);
        require(fields->states==a && fields->indices==b && fields->aux==c,
            "group64 fields differ from independent packed oracle");
        const std::array<std::size_t,4> sizes{std::size_t(range.second-range.first)*2,a.size(),b.size(),c.size()};
        std::array<std::vector<std::uint8_t>,4> direct;
        std::array<mfq::NvqRows::FieldTarget,4> targets;
        for(std::size_t f=0;f<4;++f) {
            direct[f].assign(sizes[f]+2,0xa5);targets[f]={direct[f].data()+1,sizes[f]};
        }
        std::size_t source_bytes=0;
        require(source.copy_rows_fields(range.first,range.second,targets,&source_bytes),
            "group64 could not fill direct fields");
        const std::array<const std::uint8_t*,4> expected{
            reinterpret_cast<const std::uint8_t*>(fields->neuron_scale.data()),a.data(),b.data(),c.data()};
        for(std::size_t f=0;f<4;++f)
            require(direct[f].front()==0xa5 && direct[f].back()==0xa5 &&
                std::memcmp(targets[f].data,expected[f],sizes[f])==0,
                "group64 direct fields changed packed bytes or overwrote a neighbour");
        require(source_bytes==fields->source_nbytes,"group64 direct source byte count changed");
        require(fields->source_nbytes==std::size_t(range.second-range.first)*(2+groups*8),
            "group64 source bytes included the shared codebook or expanded fields");
    }
    blob[record_offset+2]^=8; // Sign parity of the first vector.
    rejected([&]{source.slice_rows_fields(0,1);});blob[record_offset+2]^=8;
    if(vectors%3) {
        const auto byte=record_offset+std::size_t(groups-1)*8+(vectors%3)*20/8;
        const auto mask=std::uint8_t(1u<<((vectors%3)*20%8));
        blob[byte]^=mask;rejected([&]{source.slice_rows_fields(0,1);});blob[byte]^=mask;
    }
    blob[anchor_offset+1]=0x7c;
    rejected([&]{source.slice_rows_fields(0,1);});
    require(batches==10+int(vectors%3!=0),"group64 field batches were skipped");
}
void check_nint_direct_fields(int width,int gs,bool adaptive,int k=6) {
    constexpr int rows=34;
    auto fixture=mfq::test::fixture(rows,width,gs,k,adaptive);
    const int groups=(width+gs-1)/gs,padded=groups*gs;
    std::size_t read_bytes=0;
    const auto read=[&](std::size_t offset,std::uint8_t* output,std::size_t bytes) {
        require(offset<=fixture.blob.size() && bytes<=fixture.blob.size()-offset,"NINT direct read exceeded source");
        std::memcpy(output,fixture.blob.data()+offset,bytes);read_bytes+=bytes;
    };
    const mfq::NintRows source(fixture.blob.size(),read,false,[&](const std::vector<mfq::ReadSpan>& spans) {
        for(const auto& span:spans)read(span.offset,reinterpret_cast<std::uint8_t*>(span.destination),span.size);
    });
    for(const auto range:{std::pair<int,int>{0,1},{1,18},{17,34},{0,34}}) {
        const auto count=std::size_t(range.second-range.first);
        const std::array<std::size_t,7> sizes={count*std::size_t(padded)+8,count,count*8,
            count*groups,count*groups,count*2,count*2};
        std::array<std::vector<std::uint8_t>,7> fields;
        std::array<mfq::QuantFieldTarget,7> targets;
        for(std::size_t f=0;f<7;++f) {
            fields[f].assign(sizes[f]+2,0xa5);targets[f]={fields[f].data()+1,sizes[f]};
        }
        const auto selected=source.slice_rows_blob(range.first,range.second);
        mfq::copy_nint_runtime_fields(selected,targets);
        std::vector<std::uint8_t> expected_q(sizes[0],0);
        std::uint64_t qbit=0;
        for(int row=range.first;row<range.second;++row) {
            const auto local=std::size_t(row-range.first);
            const auto qb=adaptive?(row/4+row*3)%8+1:4;
            const auto kb=adaptive?k-1+(row*5+1)%4:k;
            std::int64_t offset;std::memcpy(&offset,targets[2].data+local*8,8);
            require(targets[1].data[local]==qb && offset==std::int64_t(qbit),"NINT direct q row metadata changed");
            for(int column=0;column<padded;++column) {
                const auto value=(row*17+column*11+1)% (1<<qb);
                for(int bit=0;bit<qb;++bit,++qbit)
                    expected_q[std::size_t(qbit/8)]|=((value>>bit)&1u)<<(qbit&7);
            }
            for(int group=0;group<groups;++group) {
                require(targets[3].data[local*groups+group]==(row*3+group*5+1)%(1<<kb) &&
                    targets[4].data[local*groups+group]==(row+group*7)%(1<<kb),"NINT direct subgroup metadata changed");
            }
            std::uint16_t scale,minimum;
            std::memcpy(&scale,targets[5].data+local*2,2);std::memcpy(&minimum,targets[6].data+local*2,2);
            require(scale==(row%2?0x3800:0x3c00) && minimum==0x3400,"NINT direct neuron anchors changed");
        }
        require(std::memcmp(targets[0].data,expected_q.data(),sizes[0])==0,"NINT direct packed q or its guard changed");
        for(const auto& field:fields)require(field.front()==0xa5 && field.back()==0xa5,"NINT direct copy overwrote its neighbour");
        --targets[3].bytes;rejected([&]{mfq::copy_nint_runtime_fields(selected,targets);});++targets[3].bytes;
        auto malformed=selected;malformed.pop_back();
        rejected([&]{mfq::copy_nint_runtime_fields(malformed,targets);});
        malformed=selected;malformed.push_back(0);
        rejected([&]{mfq::copy_nint_runtime_fields(malformed,targets);});
        malformed=selected;malformed[43]=0x7c;
        rejected([&]{mfq::copy_nint_runtime_fields(malformed,targets);});
    }
}

void append_pool(std::vector<std::uint8_t>& blob,const std::vector<int>& ids,
        const char* dtype,const std::vector<std::uint8_t>& body) {
    mfq::test::append<std::uint32_t>(blob,ids.size());
    mfq::test::append<std::uint32_t>(blob,std::strlen(dtype));
    mfq::test::append<std::uint64_t>(blob,body.size());
    mfq::test::append<std::uint64_t>(blob,0);
    for (int id:ids) mfq::test::append<std::int32_t>(blob,id);
    blob.insert(blob.end(),dtype,dtype+std::strlen(dtype));
    blob.insert(blob.end(),body.begin(),body.end());
}
std::vector<std::uint8_t> mixed() {
    std::vector<std::uint8_t> blob{'M','F','E','1'};
    for (unsigned value: {4,17,53,2}) mfq::test::append<std::uint32_t>(blob,value);
    append_pool(blob,{3,0},"NINT",mfq::test::fixture(34,53,5,6).blob);
    append_pool(blob,{2,1},"NVQ",nvq(0,34));
    return blob;
}
mfq::MfeQuantExpertStore memory_store(std::vector<std::uint8_t> blob,
        std::shared_ptr<std::atomic<std::size_t>> bytes=std::make_shared<std::atomic<std::size_t>>(0),bool batched=false) {
    auto source=std::make_shared<const std::vector<std::uint8_t>>(std::move(blob));
    const auto read=[source,bytes](std::size_t offset,std::uint8_t* out,std::size_t size) {
            require(offset<=source->size() && size<=source->size()-offset,"read exceeded source");
            bytes->fetch_add(size);
            std::memcpy(out,source->data()+offset,size);
        };
    mfq::NintRows::ReadBatch batch;
    if(batched)batch=[read](const std::vector<mfq::ReadSpan>& spans) {
        for(const auto& span:spans)read(span.offset,reinterpret_cast<std::uint8_t*>(span.destination),span.size);
    };
    return {source->size(),read,std::move(batch)};
}
void self_test() {
    {
        std::vector<std::uint8_t> mx{'M','X','T','1',1,4,0,0};
        for(std::uint64_t dimension:{6,64,6,32,6,2})mfq::test::append(mx,dimension);
        for(int i=0;i<192;++i)mx.push_back(std::uint8_t(i));
        for(int i=0;i<12;++i)mx.push_back(std::uint8_t(120+i));
        std::vector<std::uint8_t> blob{'M','F','E','1'};
        for(unsigned dimension:{3,3,64,2})mfq::test::append(blob,dimension);
        append_pool(blob,{2,0},"MXFP4",mx);
        append_pool(blob,{1},"NINT",mfq::test::fixture(3,64,16,6).blob);
        auto bytes=std::make_shared<std::atomic<std::size_t>>(0);
        auto store=memory_store(blob,bytes,true);
        require(bytes->load()<blob.size(),"MXFP4 registration read payloads");
        bytes->store(0);
        const auto leaf=store.read_expert(0);
        require(leaf.dtype=="MXFP4" && leaf.payload.size()==158 && bytes->load()==102,
            "MXFP4 expert read expanded or scanned the pool");
        require(std::memcmp(leaf.payload.data()+56,mx.data()+56+96,96)==0 &&
            std::memcmp(leaf.payload.data()+152,mx.data()+248+6,6)==0,"MXFP4 ownership or scales changed");
        std::array<std::vector<std::uint8_t>,2> fields{std::vector<std::uint8_t>(98,0xa5),
            std::vector<std::uint8_t>(8,0xa5)};
        std::array<mfq::QuantFieldTarget,2> targets{{{fields[0].data()+1,96},{fields[1].data()+1,6}}};
        require(store.copy_mxfp4_fields(0,targets),"MXFP4 direct fields unavailable");
        require(std::memcmp(targets[0].data,leaf.payload.data()+56,96)==0 &&
            std::memcmp(targets[1].data,leaf.payload.data()+152,6)==0 &&
            fields[0].front()==0xa5 && fields[0].back()==0xa5 &&
            fields[1].front()==0xa5 && fields[1].back()==0xa5,"MXFP4 direct field bounds changed");
        const auto untouched=fields;
        require(!store.copy_mxfp4_fields(1,targets) && fields==untouched,"NINT source overwrote MXFP4 fields");
        --targets[0].bytes;rejected([&]{store.copy_mxfp4_fields(0,targets);});
        int visited=0;store.visit_pool_experts(0,[&](int id,mfq::MfeQuantExpert expert) {
            require(expert.payload==store.read_expert(id).payload,"MXFP4 preload changed rows");++visited;
        });
        require(visited==2,"MXFP4 preload omitted experts");
        mx.back()=255;
        std::vector<std::uint8_t> bad{'M','F','E','1'};
        for(unsigned dimension:{2,3,64,1})mfq::test::append(bad,dimension);
        append_pool(bad,{0,1},"MXFP4",mx);
        auto invalid=memory_store(bad);rejected([&]{invalid.read_expert(1);});
    }
    {
        const std::vector<int> ids{5,2,7,0,6,1,4,3};
        std::vector<std::uint8_t> blob{'M','F','E','1'};
        for(unsigned value:{8,8,53,1})mfq::test::append<std::uint32_t>(blob,value);
        append_pool(blob,ids,"NVQ",nvq(0,64));
        auto store=memory_store(blob,std::make_shared<std::atomic<std::size_t>>(0),true);
        const auto first=store.read_nvq_fields(ids[1]);
        const std::array<std::size_t,4> leaf={first->neuron_scale.size()*2,first->states.size(),first->indices.size(),first->aux.size()};
        std::array<std::vector<std::uint8_t>,4> bytes;
        std::array<mfq::NvqRows::FieldTarget,4> targets;
        for(int f=0;f<4;++f) {bytes[f].assign(3*leaf[f]+2,0xa5);targets[f]={bytes[f].data()+1,3*leaf[f]};}
        const auto untouched=bytes;
        const std::vector<int> skipped{ids[1],ids[3]};
        require(!store.copy_nvq_fields(skipped,targets) && bytes==untouched,"nonconsecutive bulk experts changed destinations");
        const std::vector<int> invalid{ids[1],-1};
        rejected([&]{store.copy_nvq_fields(invalid,targets);});
        require(bytes==untouched,"invalid bulk expert changed a destination");
        const std::vector<int> selected{ids[1],ids[2],ids[3]};
        std::size_t source_bytes=0;
        require(store.copy_nvq_fields(selected,targets,&source_bytes),"consecutive pool-local experts could not share a read");
        std::size_t expected_bytes=0;
        for(std::size_t i=0;i<selected.size();++i) {
            const auto expected=store.read_nvq_fields(selected[i]);expected_bytes+=expected->source_nbytes;
            const std::array<const std::uint8_t*,4> fields={reinterpret_cast<const std::uint8_t*>(expected->neuron_scale.data()),
                expected->states.data(),expected->indices.data(),expected->aux.data()};
            for(int f=0;f<4;++f)require(std::memcmp(bytes[f].data()+1+i*leaf[f],fields[f],leaf[f])==0,
                "bulk NVQ fields differ from independent expert slices");
        }
        for(const auto& field:bytes)require(field.front()==0xa5 && field.back()==0xa5,"bulk NVQ copy overwrote guards");
        require(source_bytes==expected_bytes,"bulk NVQ source byte accounting changed");
    }
    for(int bits=1;bits<=8;++bits)check_nint_direct_fields(53,5,false,bits);
    for(int bits:{2,4})check_nint_direct_fields(640,48,true,bits);
    for(int width:{1,7,8,9,23,24,25,53,64,127,128,191,192,193,200,640,2560})check_group64_fields(width);
    for(bool adaptive:{false,true})for(int width:{53,128,640})for(int gs:{5,21,24,28})
        check_nint_direct_fields(width,gs,adaptive);
    const auto flat = mfq::test::fixture(52, 53, 5, 6).blob;
    auto packed = std::make_shared<std::vector<std::uint8_t>>(flat.begin(), flat.begin() + 14);
    mfq::test::append<std::uint32_t>(*packed, 3);
    for (std::int64_t dimension : {4, 13, 53}) mfq::test::append<std::int64_t>(*packed, dimension);
    packed->insert(packed->end(), flat.begin() + 34, flat.end());
    auto packed_reads = std::make_shared<std::size_t>(0);
    auto read_packed = [packed, packed_reads](std::size_t offset, std::uint8_t* out, std::size_t count) {
        require(offset <= packed->size() && count <= packed->size() - offset, "packed NINT read exceeded source");
        *packed_reads += count;
        std::memcpy(out, packed->data() + offset, count);
    };
    const auto packed_store = mfq::make_packed_nint_expert_store(packed->size(), read_packed);
    require(packed_store->num_experts() == 4 && packed_store->out_per_expert() == 13 && packed_store->neuron_len() == 53,
        "packed NINT expert geometry changed");
    require(*packed_reads < packed->size() / 4, "packed NINT index read weight payloads");
    int packed_batches=0;
    const auto packed_batch_store=mfq::make_packed_nint_expert_store(packed->size(),read_packed,
        [&](const std::vector<mfq::ReadSpan>& spans) {
            ++packed_batches;require(spans.size()>2,"packed NINT did not batch its independent fields");
            for(const auto& span:spans)read_packed(span.offset,reinterpret_cast<std::uint8_t*>(span.destination),span.size);
        });
    mfq::NintRows flat_rows(flat.data(), flat.size());
    for (int expert = 0; expert < 4; ++expert) {
        require(packed_store->read_expert(expert).payload == flat_rows.slice_rows_blob(expert * 13, (expert + 1) * 13),
            "packed NINT expert bytes changed");
        require(packed_batch_store->read_expert(expert).payload==packed_store->read_expert(expert).payload,
            "packed NINT batched field offsets or rebasing changed bytes");
    }
    require(packed_batches==4,"packed NINT did not read each expert's fields in one batch");
    rejected([&] { mfq::make_packed_nint_expert_store(49, read_packed); });
    (*packed)[14] = 2;
    rejected([&] { mfq::make_packed_nint_expert_store(packed->size(), read_packed); });
    (*packed)[14] = 3;

    const auto original=mixed();
    auto bytes=std::make_shared<std::atomic<std::size_t>>(0);
    const auto store=memory_store(original,bytes);
    const auto batched=memory_store(original,std::make_shared<std::atomic<std::size_t>>(0),true);
    require(store.num_experts()==4 && store.out_per_expert()==17 && store.neuron_len()==53,
        "mixed expert geometry changed");
    require(bytes->load()<original.size()/4 && store.index_nbytes()<original.size()/4,
        "mixed expert index read or retained weight payloads");
    const auto reference=mfq::test::fixture(34,53,5,6);
    for (int expert: {0,1,2,3}) {
        bytes->store(0);
        const auto budget=store.expert_payload_nbytes(expert);
        require(bytes->load()==0,"expert byte budget read a weight payload");
        const auto result=store.read_expert(expert);
        require(result.payload.size()==budget,"expert payload exceeded its byte budget");
        require(bytes->load()<original.size()/2,"one expert read unrelated payloads");
        const auto fields=store.read_nvq_fields(expert);
        const auto bulk=batched.read_nvq_fields(expert);
        require(bool(fields)==bool(bulk),"batched expert family changed");
        if(fields)require(fields->neuron_scale==bulk->neuron_scale && fields->states==bulk->states &&
            fields->indices==bulk->indices && fields->aux==bulk->aux,"batched expert bits changed");
        if (expert==0 || expert==3) {
            require(!fields,"NINT expert was exposed as NVQ fields");
            require(result.dtype=="NINT","mixed expert dtype changed");
            mfq::NintRows rows(result.payload.data(),result.payload.size());
            mfq::NintRowBatch decoded;
            for (int row=0; row<17; ++row) rows.append_row(row,decoded);
            const int begin=expert==3 ? 0 : 17;
            for (int row=0; row<17; ++row)
                for (int column=0; column<53; ++column)
                    require(mfq::test::row_value(decoded,row,column)==reference.reference[(row+begin)*53+column],
                        "adaptive NINT expert differs from independent code oracle");
        } else {
            const auto expected=nvq(expert==2 ? 0 : 17,expert==2 ? 17 : 34);
            require(result.dtype=="NVQ" && result.payload==expected,"NVQ expert packed fields or padding changed");
            require(bool(fields),"canonical NVQ fields were unavailable");
            std::vector<std::uint8_t> direct(expected.begin(),expected.begin()+40);
            const auto* anchors=reinterpret_cast<const std::uint8_t*>(fields->neuron_scale.data());
            direct.insert(direct.end(),anchors,anchors+fields->neuron_scale.size()*2);
            for(const auto* stream:{&fields->states,&fields->indices,&fields->aux})
                direct.insert(direct.end(),stream->begin(),stream->end());
            require(direct==expected && fields->nbytes()+40==expected.size(),
                "direct NVQ fields changed non-byte-aligned expert data");
        }
    }
    std::vector<std::future<std::vector<std::uint8_t>>> concurrent;
    for (std::size_t pool=0; pool<store.pool_count(); ++pool) {
        std::vector<std::pair<int,mfq::MfeQuantExpert>> leaves;
        bytes->store(0);
        store.visit_pool_experts(pool,[&](int expert,mfq::MfeQuantExpert encoded) {
            leaves.emplace_back(expert,std::move(encoded));
        });
        require(leaves.size()==store.pool_expert_ids(pool).size(),"startup pool expert coverage changed");
        const auto pool_read_bytes=bytes->load();
        require(pool_read_bytes<original.size(),"startup pool read crossed pool bounds");
        for (std::size_t i=0; i<leaves.size(); ++i) {
            const auto expected=store.read_expert(leaves[i].first);
            require(leaves[i].first==store.pool_expert_ids(pool)[i] &&
                leaves[i].second.dtype==expected.dtype && leaves[i].second.payload==expected.payload,
                "startup sequential pool differs from canonical independent expert slices");
        }
        mfq::HostParallelPool workers({},false);
        std::vector<mfq::MfeQuantExpert> parallel(leaves.size());
        const auto visit=[&](int expert,mfq::MfeQuantExpert encoded) {
            const auto& ids=store.pool_expert_ids(pool);
            const auto row=std::find(ids.begin(),ids.end(),expert)-ids.begin();
            parallel.at(row)=std::move(encoded);
        };
        store.visit_pool_experts(pool,visit,&workers,4);
        for(std::size_t i=0;i<leaves.size();++i)
            require(parallel[i].dtype==leaves[i].second.dtype && parallel[i].payload==leaves[i].second.payload,
                "parallel startup slicing changed canonical expert bytes");
        rejected([&] {store.visit_pool_experts(pool,[](int,mfq::MfeQuantExpert) {
            throw std::runtime_error("parallel visitor failed");
        },&workers,4);});
        store.visit_pool_experts(pool,visit,&workers,4);
    }
    rejected([&] { store.visit_pool_experts(store.pool_count(),[](int,mfq::MfeQuantExpert) {}); });
    rejected([&] { store.visit_pool_experts(0,{}); });
    rejected([&] { store.visit_pool_experts(0,[](int,mfq::MfeQuantExpert) { throw std::runtime_error("visitor failed"); }); });
    for (int i=0; i<12; ++i) concurrent.push_back(std::async(std::launch::async,[&store,i] {
        return store.read_expert(i%4).payload;
    }));
    for (int i=0; i<12; ++i) require(concurrent[i].get()==store.read_expert(i%4).payload,
        "concurrent expert reads changed bytes");
    rejected([&] { store.read_expert(-1); }); rejected([&] { store.read_expert(4); });
    rejected([&] { store.read_nvq_fields(-1); }); rejected([&] { store.read_nvq_fields(4); });
    rejected([&] { store.expert_dtype(4); });
    for (int kind=0; kind<7; ++kind) {
        auto bad=original;
        if (kind==0) bad[0]='X';
        if (kind==1) bad.pop_back();
        if (kind==2) bad.push_back(0);
        if (kind==3) bad[36]=1; // Unsupported embedded runtime field.
        if (kind==4) bad[48]=3; // Duplicate global expert ownership.
        if (kind==5) bad[12]=54; // Pool and logical input shapes disagree.
        if (kind==6) bad[24]=0; // Empty pool dtype.
        rejected([&] { memory_store(std::move(bad)); });
    }
    const auto single=nvq(0,34);
    const auto nvq_reader=[&](std::size_t offset,std::uint8_t* out,std::size_t size) {
        require(offset<=single.size() && size<=single.size()-offset,"NVQ read exceeded source");
        std::memcpy(out,single.data()+offset,size);
    };
    const mfq::NvqRows rows(single.size(),nvq_reader);
    for(const bool batch:{false,true}) {
        mfq::NvqRows::ReadBatch read_batch;
        if(batch)read_batch=[&](const std::vector<mfq::ReadSpan>& spans) {
            require(spans.size()==4,"direct NVQ fields were not read together");
            for(const auto& span:spans)nvq_reader(span.offset,reinterpret_cast<std::uint8_t*>(span.destination),span.size);
        };
        const mfq::NvqRows direct(single.size(),nvq_reader,read_batch);
        const auto expected=nvq(8,16);
        std::array<std::vector<std::uint8_t>,4> buffers;
        std::array<mfq::NvqRows::FieldTarget,4> targets;
        const std::array<std::size_t,4> sizes={16,9,77,3};
        std::size_t cursor=40;
        for(int f=0;f<4;++f) {
            buffers[f].resize(sizes[f]+8,0xa5);targets[f]={buffers[f].data()+3,sizes[f]};
        }
        require(direct.copy_rows_fields(8,16,targets),"aligned NVQ direct fields were unavailable");
        for(int f=0;f<4;++f) {
            require(std::equal(buffers[f].begin()+3,buffers[f].end()-5,expected.begin()+cursor),
                "direct NVQ target differs from independent encoding");
            require(std::all_of(buffers[f].begin(),buffers[f].begin()+3,[](auto value){return value==0xa5;}) &&
                std::all_of(buffers[f].end()-5,buffers[f].end(),[](auto value){return value==0xa5;}),
                "direct NVQ fields overwrote a target guard");
            cursor+=sizes[f];
        }
        require(!direct.copy_rows_fields(1,9,targets),"unaligned NVQ target copy was accepted");
        --targets[1].bytes;rejected([&]{direct.copy_rows_fields(8,16,targets);});
    }
    for(const std::uint16_t anchor:{0x7c00,0x7e01,0x8000,0xbc00}) {
        auto bad=single;std::memcpy(bad.data()+40,&anchor,2);
        mfq::NvqRows invalid(bad.size(),[&](std::size_t offset,std::uint8_t* output,std::size_t count) {
            std::memcpy(output,bad.data()+offset,count);
        });
        rejected([&]{invalid.slice_rows_fields(0,1);});
    }
    require(rows.slice_rows_blob(0,34)==single,"full NVQ slice changed bytes");
    require(rows.row_range_nbytes(0,34)==single.size(),"NVQ range byte budget mismatch");
    rejected([&] { rows.slice_rows_blob(-1,1); });
    rejected([&] { rows.slice_rows_blob(17,17); });
    rejected([&] { rows.slice_rows_blob(0,35); });
    for (int kind=0; kind<8; ++kind) {
        auto bad=single;
        if (kind==0) bad[0]='X';
        if (kind==1) bad.pop_back();
        if (kind==2) bad.push_back(0);
        if (kind==3) bad[4]=3;
        if (kind==4) bad[16]=3;
        if (kind==5) bad[6]=25;
        if (kind==6) bad[36]=0;
        if (kind==7) bad[5]=9;
        rejected([&] { mfq::NvqRows invalid(bad.size(),[&](std::size_t offset,std::uint8_t* out,std::size_t size) {
            require(offset<=bad.size() && size<=bad.size()-offset,"invalid NVQ read exceeded source");
            std::memcpy(out,bad.data()+offset,size);
        }); });
    }
    std::cout << "mixed quant expert range self-test passed\n";
}
void index_model(const char* path) {
    const auto source=mfq::open_model_source(path);
    const auto start=std::chrono::steady_clock::now();
    std::uint64_t bytes=0,calls=0,index=0,payload=0,experts=0;
    std::vector<std::unique_ptr<mfq::MfeQuantExpertStore>> stores;
    for (const auto& tensor:source->tensors()) {
        if (tensor.dtype!="MFE") continue;
        const auto reader=source->tensor_reader(tensor.name);
        auto store=std::make_unique<mfq::MfeQuantExpertStore>(tensor.nbytes,
            [reader,&bytes,&calls](std::size_t offset,std::uint8_t* out,std::size_t size) {
                reader(offset,reinterpret_cast<std::byte*>(out),size);
                bytes+=size; ++calls;
            });
        index+=store->index_nbytes(); payload+=tensor.nbytes; experts+=store->num_experts();
        stores.push_back(std::move(store));
    }
    require(!stores.empty(),"no MFE tensors indexed");
    std::cout << "model_range_tensors=" << stores.size() << " experts=" << experts
        << " range_index_bytes=" << index << " index_read_bytes=" << bytes << " index_calls=" << calls
        << " tensor_bytes=" << payload << " wall_ms="
        << std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count() << '\n';
}
void dump(int argc,char** argv) {
    require(argc>=5,"expected MODEL TENSOR OUTPUT_DIRECTORY EXPERT_ID...");
    auto source=mfq::open_model_source(argv[1]);
    const auto* metadata=source->find_tensor(argv[2]);
    require(metadata!=nullptr,"mixed tensor not found");
    std::uint64_t bytes=0,calls=0;
    const auto reader=source->tensor_reader(argv[2]);
    require(metadata->dtype=="MFE","expected MFE tensor dtype");
    const auto start=std::chrono::steady_clock::now();
    const mfq::MfeQuantExpertStore store(metadata->nbytes,
        [reader,&bytes,&calls](std::size_t offset,std::uint8_t* out,std::size_t size) {
            reader(offset,reinterpret_cast<std::byte*>(out),size);
            bytes+=size; ++calls;
        });
    const auto index_bytes_read=bytes,index_calls=calls;
    source.reset(); // The independently owned reader must remain valid.
    std::filesystem::create_directories(argv[3]);
    std::ofstream manifest(std::filesystem::path(argv[3])/"experts.txt");
    require(bool(manifest),"could not create required expert manifest");
    for (int i=4; i<argc; ++i) {
        const int expert=std::stoi(argv[i]); bytes=0; calls=0;
        const auto result=store.read_expert(expert);
        std::ofstream output(std::filesystem::path(argv[3])/("expert-"+std::to_string(expert)+".payload"),std::ios::binary);
        output.write(reinterpret_cast<const char*>(result.payload.data()),result.payload.size());
        output.close(); require(bool(output),"could not write required expert payload");
        manifest << expert << ' ' << result.dtype << ' ' << result.payload.size() << ' '
            << bytes << ' ' << calls << '\n';
        std::cout << "expert=" << expert << " dtype=" << result.dtype << " bytes=" << result.payload.size()
            << " read_bytes=" << bytes << " calls=" << calls << '\n';
    }
    manifest.close(); require(bool(manifest),"could not write required expert manifest");
    std::cout << "range_index_bytes=" << store.index_nbytes() << " index_read_bytes=" << index_bytes_read
        << " index_calls=" << index_calls << " tensor_bytes=" << store.payload_nbytes()
        << " wall_ms=" << std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count() << '\n';
}
}
int main(int argc,char** argv) try {

    if (argc==1) self_test();
    else if (argc==3 && std::string(argv[1])=="--index-model") index_model(argv[2]);
    else dump(argc,argv);
    return 0;
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
