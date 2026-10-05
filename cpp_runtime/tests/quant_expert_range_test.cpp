#include "mfq/mfe_quant_expert_store.h"
#include "mfq/model_source.h"
#include "nint_row_fixture.h"

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
        std::shared_ptr<std::atomic<std::size_t>> bytes=std::make_shared<std::atomic<std::size_t>>(0)) {
    auto source=std::make_shared<const std::vector<std::uint8_t>>(std::move(blob));
    return {source->size(),
        [source,bytes](std::size_t offset,std::uint8_t* out,std::size_t size) {
            require(offset<=source->size() && size<=source->size()-offset,"read exceeded source");
            bytes->fetch_add(size);
            std::memcpy(out,source->data()+offset,size);
        }};
}
void self_test() {
    const auto original=mixed();
    auto bytes=std::make_shared<std::atomic<std::size_t>>(0);
    const auto store=memory_store(original,bytes);
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
        if (expert==0 || expert==3) {
            require(result.dtype=="NINT","mixed expert dtype changed");
            mfq::NintRows rows(result.payload.data(),result.payload.size());
            mfq::NintRowBatch decoded;
            for (int row=0; row<17; ++row) rows.append_row(row,decoded);
            const int begin=expert==3 ? 0 : 17;
            for (int row=0; row<17; ++row)
                for (int column=0; column<53; ++column)
                    require(mfq::test::row_value(decoded,row,column)==reference.reference[(row+begin)*53+column],
                        "adaptive NINT expert differs from independent code oracle");
        } else require(result.dtype=="NVQ" && result.payload==nvq(expert==2 ? 0 : 17,expert==2 ? 17 : 34),
            "NVQ expert packed fields or padding changed");
    }
    std::vector<std::future<std::vector<std::uint8_t>>> concurrent;
    for (int i=0; i<12; ++i) concurrent.push_back(std::async(std::launch::async,[&store,i] {
        return store.read_expert(i%4).payload;
    }));
    for (int i=0; i<12; ++i) require(concurrent[i].get()==store.read_expert(i%4).payload,
        "concurrent expert reads changed bytes");
    rejected([&] { store.read_expert(-1); }); rejected([&] { store.read_expert(4); });
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
