#include "mfq/file_range_reader.h"
#include "mfq/async_range_reader.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace {
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
void exercise_large_batches(const std::filesystem::path& base) {
    const auto path=std::filesystem::path(base.string()+"-large-batch");
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup(){std::error_code ignored;std::filesystem::remove(path,ignored);}
    } cleanup{path};
    constexpr std::size_t mib=1u<<20;
    std::vector<std::byte> expected(20*mib+4096);
    for(std::size_t i=0;i<expected.size();++i)expected[i]=std::byte((i*37+i/257)&255);
    {std::ofstream stream(path,std::ios::binary);
        stream.write(reinterpret_cast<const char*>(expected.data()),expected.size());
        require(bool(stream),"cannot write large batch fixture");}
    mfq::FileRangeReader reader(path,mfq::FileReadMode::Direct);
    const std::array<std::size_t,3> offsets{3,37,7*mib+5};
    const std::array<std::size_t,3> sizes{17,11*mib+17,9*mib+33};
    std::array<std::vector<std::byte>,3> output;
    std::vector<mfq::ReadSpan> spans;
    for(std::size_t i=0;i<sizes.size();++i) {
        output[i].resize(sizes[i]+2);
        spans.push_back({offsets[i],output[i].data()+1,sizes[i]});
    }
    std::uint64_t allocations=0;
    for(int round=0;round<2;++round) {
        for(auto& field:output)std::fill(field.begin(),field.end(),std::byte{0x5a});
        reader.read_batch(spans);
        for(std::size_t i=0;i<sizes.size();++i) {
            require(output[i].front()==std::byte{0x5a} && output[i].back()==std::byte{0x5a},
                "large batch wave overwrote destination guards");
            require(std::equal(output[i].begin()+1,output[i].end()-1,expected.begin()+offsets[i]),
                "large batch wave changed field bytes");
        }
        const auto stats=reader.stats();
        require(!stats.staging_bytes && stats.staging_peak_bytes<=std::max<std::size_t>(8*mib,reader.alignment()),
            "large batch exceeded reusable staging capacity");
        if(!round)allocations=stats.staging_allocations;
        else require(stats.staging_allocations==allocations,
            "large batch discarded its staging arena between reads");
    }
}
void exercise_coalesced_staging(const std::filesystem::path& base) {
    const auto path=std::filesystem::path(base.string()+"-coalesced");
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup(){std::error_code ignored;std::filesystem::remove(path,ignored);}
    } cleanup{path};
    constexpr std::size_t mib=1u<<20;
    std::vector<std::byte> expected(3*mib+4096);
    for(std::size_t i=0;i<expected.size();++i)expected[i]=std::byte((i*19+i/137)&255);
    {std::ofstream stream(path,std::ios::binary);
        stream.write(reinterpret_cast<const char*>(expected.data()),expected.size());
        require(bool(stream),"cannot write coalesced staging fixture");}
    mfq::AsyncRangeReader reader(path,mfq::FileReadMode::Direct);
    const std::array<std::size_t,3> offsets{3,mib+20,2*mib+11};
    const std::array<std::size_t,3> sizes{mib+17,mib-9,mib-1};
    std::array<std::vector<std::byte>,3> output;
    std::vector<mfq::ReadSpan> spans;
    for(std::size_t i=0;i<output.size();++i) {
        output[i].assign(sizes[i]+2,std::byte{0x5a});
        spans.push_back({offsets[i],output[i].data()+1,sizes[i]});
    }
    reader.read(spans);
    for(std::size_t i=0;i<output.size();++i) {
        require(output[i].front()==std::byte{0x5a} && output[i].back()==std::byte{0x5a},
                "coalesced read overwrote a field guard");
        require(std::equal(output[i].begin()+1,output[i].end()-1,expected.begin()+offsets[i]),
                "coalesced read changed field bytes");
    }
    const auto stats=reader.stats();
    require(stats.logical_bytes==sizes[0]+sizes[1]+sizes[2] && !stats.staging_bytes,
            "coalesced read leaked staging or changed logical byte accounting");
    require(stats.staging_peak_bytes<=expected.size()+(64u<<10),
            "coalesced read retained two full-sized payload buffers");
    const auto tail_path=std::filesystem::path(base.string()+"-coalesced-tail");
    Cleanup tail_cleanup{tail_path};
    const auto tail_size=3*mib+19;
    {std::ofstream stream(tail_path,std::ios::binary);
        stream.write(reinterpret_cast<const char*>(expected.data()),tail_size);
        require(bool(stream),"cannot write coalesced EOF fixture");}
    for(auto mode:{mfq::FileReadMode::Buffered,mfq::FileReadMode::Direct}) {
        mfq::AsyncRangeReader tail(tail_path,mode);
        std::vector<std::byte> first(mib+19,std::byte{0x5a});
        std::vector<std::byte> second(tail_size-mib-18,std::byte{0x5a});
        tail.read({{3,first.data()+1,mib+17},{mib+20,second.data()+1,tail_size-mib-20}});
        require(first.front()==std::byte{0x5a} && first.back()==std::byte{0x5a} &&
                second.front()==std::byte{0x5a} && second.back()==std::byte{0x5a},
                "coalesced EOF read overwrote field guards");
        require(std::equal(first.begin()+1,first.end()-1,expected.begin()+3) &&
                std::equal(second.begin()+1,second.end()-1,expected.begin()+mib+20),
                "coalesced EOF read changed bytes");
        require(tail.stats().staging_peak_bytes<=tail_size+(128u<<10),
                "unaligned EOF retained another full coalesced payload");
        mfq::FileRangeReader grouped(path,mode);
        std::vector<std::byte> actual(4002,std::byte{0x5a}),reference=actual;
        std::copy_n(expected.begin()+100,3000,reference.begin()+1);
        std::copy_n(expected.begin()+200,3000,reference.begin()+1001);
        grouped.read_coalesced({{100,actual.data()+1,3000},{200,actual.data()+1001,3000}});
        require(actual==reference && grouped.stats().logical_bytes==6000,
                "coalesced reads changed caller order or duplicate-byte accounting");
    }
    mfq::FileRangeReader aligned(path,mfq::FileReadMode::Direct);
    const auto alignment=aligned.alignment();
    struct AlignedOutput {
        std::size_t alignment;
        std::byte* data;
        ~AlignedOutput(){::operator delete(data,std::align_val_t{alignment});}
    } direct{alignment,static_cast<std::byte*>(::operator new(4*alignment,std::align_val_t{alignment}))};
    std::fill_n(direct.data,4*alignment,std::byte{0x5a});
    aligned.read_batch({{2*alignment,direct.data+alignment,alignment},{0,direct.data+2*alignment,alignment}});
    require(std::equal(direct.data+alignment,direct.data+2*alignment,expected.begin()+2*alignment) &&
            std::equal(direct.data+2*alignment,direct.data+3*alignment,expected.begin()),
            "aligned direct batch changed field bytes");
    require(std::all_of(direct.data,direct.data+alignment,[](auto value){return value==std::byte{0x5a};}) &&
            std::all_of(direct.data+3*alignment,direct.data+4*alignment,[](auto value){return value==std::byte{0x5a};}) &&
            !aligned.stats().staging_peak_bytes,
            "aligned direct batch copied outside its target or allocated bounce storage");
}
void exercise_shared_staging(const std::filesystem::path& base) {
    struct Fixtures {
        std::array<std::filesystem::path,2> paths;
        ~Fixtures() {
            for(const auto& path:paths) {
                std::error_code ignored;
                std::filesystem::remove(path,ignored);
            }
        }
    } fixtures{{base.string()+"-shard-a",base.string()+"-shard-b"}};
    const std::size_t bytes=(6u<<20)+17;
    std::array<std::vector<std::byte>,2> expected;
    for(std::size_t shard=0;shard<2;++shard) {
        expected[shard].resize(bytes+4096);
        for(std::size_t i=0;i<expected[shard].size();++i)
            expected[shard][i]=std::byte((i*37+i/113+shard*97)&255);
        std::ofstream stream(fixtures.paths[shard],std::ios::binary);
        stream.write(reinterpret_cast<const char*>(expected[shard].data()),expected[shard].size());
        require(bool(stream),"cannot write shared staging fixture");
    }
    auto check=[&](mfq::FileRangeReader& reader,std::size_t shard,std::size_t offset) {
        std::vector<std::byte> actual(bytes+2,std::byte{0x5a});
        reader.read(offset,actual.data()+1,bytes);
        require(actual.front()==std::byte{0x5a} && actual.back()==std::byte{0x5a},
                "shared staging overwrote destination guards");
        require(std::equal(actual.begin()+1,actual.end()-1,expected[shard].begin()+offset),
                "shared staging reused another shard's contents");
        require(!reader.stats().staging_bytes,"shared staging retained an active read");
    };
    auto first=std::make_unique<mfq::FileRangeReader>(fixtures.paths[0],mfq::FileReadMode::Direct);
    check(*first,0,3);
    require(first->stats().staging_cached_bytes>=bytes,"large expert staging was not cached");
    {
        mfq::FileRangeReader second(fixtures.paths[1],mfq::FileReadMode::Direct);
        check(second,1,5);
        require(!second.stats().staging_allocations && !first->stats().staging_cached_bytes &&
                second.stats().staging_cached_bytes>=bytes,
                "large staging was not shared or its ownership accounting was duplicated");
    }
    const auto allocations=first->stats().staging_allocations;
    check(*first,0,7);
    require(first->stats().staging_allocations==allocations,
            "closing the previous reader discarded reusable shared staging");
    std::vector<std::unique_ptr<mfq::FileRangeReader>> readers;
    for(int i=0;i<12;++i)
        readers.push_back(std::make_unique<mfq::FileRangeReader>(fixtures.paths[i%2],mfq::FileReadMode::Direct));
    std::vector<std::future<void>> jobs;
    for(int i=0;i<12;++i)jobs.push_back(std::async(std::launch::async,[&,i] {
        for(int round=0;round<3;++round)check(*readers[i],i%2,11+i+round);
    }));
    for(auto& job:jobs)job.get();
    auto cached=first->stats().staging_cached_bytes;
    for(const auto& reader:readers)cached+=reader->stats().staging_cached_bytes;
    require(cached>0 && cached<=(128u<<20),"shared staging exceeded its process-wide idle budget");
}
void exercise(const std::filesystem::path& path, const std::vector<std::byte>& expected, mfq::FileReadMode mode) {
    auto reader = std::make_shared<mfq::FileRangeReader>(path, mode);
    const auto alignment = reader->alignment();
    if (mode == mfq::FileReadMode::Direct) require(alignment > 1, "direct reader did not query sector alignment");
    const std::vector<std::size_t> offsets = {0,1,17,511,512,4095,4096,4097,expected.size()-7};
    for (auto offset : offsets) {
        for (auto count : {std::size_t(1),std::size_t(7),std::size_t(4097),expected.size()-offset}) {
            if (count > expected.size()-offset) continue;
            std::vector<std::byte> actual(count+2,std::byte{0x5a});
            reader->read(offset,actual.data()+1,count);
            require(actual.front()==std::byte{0x5a} && actual.back()==std::byte{0x5a}, "range read overwrote destination guards");
            require(std::equal(actual.begin()+1,actual.end()-1,expected.begin()+offset), "sector/tail range bytes differ");
        }
    }
    reader->read(expected.size(),nullptr,0);
    bool rejected = false;
    try { reader->read(expected.size()-1,nullptr,2); } catch (const std::out_of_range&) { rejected=true; }
    require(rejected,"file bounds were not checked before destination");
    rejected=false;
    try { reader->read(0,nullptr,1); } catch (const std::invalid_argument&) { rejected=true; }
    require(rejected,"null destination was accepted");
    std::vector<std::future<void>> workers;
    for (int worker=0;worker<12;++worker) {
        workers.push_back(std::async(std::launch::async,[reader,&expected,worker] {
            for (int repeat=0;repeat<32;++repeat) {
                const auto offset=(std::size_t(worker)*4919+repeat*271)% (expected.size()-8193);
                std::vector<std::byte> actual(8193);
                reader->read(offset,actual.data(),actual.size());
                require(std::equal(actual.begin(),actual.end(),expected.begin()+offset),"concurrent positional reads raced");
            }
        }));
    }
    for (auto& worker:workers) worker.get();
    const auto stats=reader->stats();
    require(stats.files==1 && stats.mode==mode && stats.calls>=384 && stats.errors==0,"read accounting differs");
    require(stats.logical_bytes>0 && stats.physical_bytes>=stats.logical_bytes,"read byte accounting differs");
    require(stats.staging_bytes==0,"aligned staging leaked after read");
    if (mode==mfq::FileReadMode::Direct) require(stats.staging_peak_bytes>=alignment,"aligned staging was not charged");
    else require(stats.staging_peak_bytes==0,"buffered reader allocated unnecessary staging");
}
}

int main() {
    const auto nonce=std::chrono::steady_clock::now().time_since_epoch().count();
    const auto path=std::filesystem::temp_directory_path()/std::filesystem::u8path("mfq-file-读取-"+std::to_string(nonce)+".bin");
    try {
        std::vector<std::byte> expected(65543);
        for (std::size_t i=0;i<expected.size();++i) expected[i]=static_cast<std::byte>((i*73+i/131)&255);
        { std::ofstream stream(path,std::ios::binary); stream.write(reinterpret_cast<const char*>(expected.data()),expected.size());
          require(bool(stream),"cannot write fixture"); }
        exercise_coalesced_staging(path);
        exercise_large_batches(path);
        {
            const auto sector=mfq::FileRangeReader(path,mfq::FileReadMode::Direct).alignment();
            std::vector<std::byte> large(256*sector+13);
            for(std::size_t i=0;i<large.size();++i)large[i]=std::byte((i*31+i/19)&255);
            {std::ofstream stream(path,std::ios::binary|std::ios::trunc);
                stream.write(reinterpret_cast<const char*>(large.data()),large.size());require(bool(stream),"cannot write sparse row fixture");}
            {
                mfq::AsyncRangeReader rows(path,mfq::FileReadMode::Direct);
                std::vector<std::byte> output(96*9,std::byte{0x5a});std::vector<mfq::ReadSpan> spans;
                for(std::size_t i=0;i<96;++i)spans.push_back({i*2*sector+3,output.data()+i*9+1,7});
                rows.read(spans);
                for(std::size_t i=0;i<96;++i) {
                    require(output[i*9]==std::byte{0x5a} && output[i*9+8]==std::byte{0x5a},"sparse coalescing changed guards");
                    require(std::equal(output.begin()+i*9+1,output.begin()+i*9+8,large.begin()+i*2*sector+3),"sparse coalescing changed row bytes");
                }
                const auto stats=rows.stats();
                require((sector>65536/3 || stats.calls<96) && stats.physical_bytes<=2*96*sector && !stats.staging_bytes,
                    "sparse rows did not coalesce or exceeded their physical read bound");
            }
            {std::ofstream stream(path,std::ios::binary|std::ios::trunc);
                stream.write(reinterpret_cast<const char*>(expected.data()),expected.size());require(bool(stream),"cannot restore fixture");}
        }
        exercise(path,expected,mfq::FileReadMode::Buffered);
        exercise(path,expected,mfq::FileReadMode::Direct);
        for(auto mode:{mfq::FileReadMode::Buffered,mfq::FileReadMode::Direct}) {
            {
                mfq::AsyncRangeReader disjoint(path,mode);
                std::vector<std::byte> output(8*9,std::byte{0x5a});std::vector<mfq::ReadSpan> spans;
                for(std::size_t i=0;i<8;++i)spans.push_back({i*8192+3,output.data()+i*9+1,7});
                disjoint.read(spans);
                for(std::size_t i=0;i<8;++i) {
                    require(output[i*9]==std::byte{0x5a} && output[i*9+8]==std::byte{0x5a},"disjoint read changed guards");
                    require(std::equal(output.begin()+i*9+1,output.begin()+i*9+8,expected.begin()+i*8192+3),"disjoint read changed bytes");
                }
                const auto stats=disjoint.stats();
                require(stats.logical_bytes==56 && !stats.staging_bytes && stats.staging_peak_bytes<=stats.physical_bytes,
                    "disjoint direct ranges allocated a redundant staging copy");
            }
            {
                mfq::FileRangeReader batched(path,mode);
                std::uint64_t first_allocations=0;
                for(int round=0;round<4;++round) {
                    std::vector<std::byte> output(384*33,std::byte{0x5a});std::vector<mfq::ReadSpan> spans;
                    for(std::size_t i=0;i<384;++i)spans.push_back({i*167+std::size_t(round),output.data()+i*33+1,31});
                    batched.read_batch(spans);
                    if(!round)first_allocations=batched.stats().staging_allocations;
                    if(mode==mfq::FileReadMode::Direct)require(batched.stats().staging_allocations==first_allocations &&
                        batched.stats().staging_cached_bytes<=(8u<<20),"repeated Direct batches did not reuse bounded aligned arenas");
                    for(std::size_t i=0;i<384;++i) {
                        require(output[i*33]==std::byte{0x5a} && output[i*33+32]==std::byte{0x5a},"event reuse changed a destination guard");
                        require(std::equal(output.begin()+i*33+1,output.begin()+i*33+32,expected.begin()+i*167+round),
                            "event reuse or a batch beyond its event ring changed bytes");
                    }
                }
                require(!batched.stats().staging_bytes && !batched.stats().errors,"event reuse leaked request staging");
                if(mode==mfq::FileReadMode::Direct)
                    require(batched.stats().staging_peak_bytes<=128*batched.alignment(),
                        "direct row batches allocated unbounded aligned staging");
            }
            {
                mfq::FileRangeReader growing(path,mode);
                std::vector<std::vector<std::byte>> output;
                std::vector<mfq::ReadSpan> spans;
                std::size_t logical=0;
                for(std::size_t i=0;i<192;++i) {
                    const std::size_t bytes=i<64?31:(i<128?8193:7);
                    const auto offset=(i*997)%(expected.size()-bytes+1);
                    output.emplace_back(bytes+2,std::byte{0x5a});
                    spans.push_back({offset,output.back().data()+1,bytes});logical+=bytes;
                }
                growing.read_batch(spans);
                for(std::size_t i=0;i<spans.size();++i) {
                    require(output[i].front()==std::byte{0x5a} && output[i].back()==std::byte{0x5a},"growing batch arena overwrote guards");
                    require(std::equal(output[i].begin()+1,output[i].end()-1,expected.begin()+spans[i].offset),
                        "growing or reusing a batch arena changed bytes");
                }
                const auto stats=growing.stats();
                require(!stats.staging_bytes && !stats.errors && stats.logical_bytes==logical,"growing batch staging accounting differs");
                if(mode==mfq::FileReadMode::Direct)
                    require(stats.staging_peak_bytes<=64*(8193+2*growing.alignment()),"growing batch retained more than one wave");
            }
            auto reader=std::make_shared<mfq::AsyncRangeReader>(path,mode);
            std::vector<std::future<void>> jobs;
            for(int client=0;client<4;++client)jobs.push_back(std::async(std::launch::async,[&,reader,client] {
                for(int round=0;round<8;++round) {
                    const std::vector<std::size_t> offsets={1,4095,4096,4097,expected.size()-7,std::size_t(client*73+round*111)};
                    std::vector<std::vector<std::byte>> buffers;
                    for(auto at:offsets)buffers.emplace_back(std::min<std::size_t>(4177,expected.size()-at)+2,std::byte{0x5a});
                    std::vector<mfq::ReadSpan> spans;
                    for(std::size_t i=0;i<offsets.size();++i)spans.push_back({offsets[i],buffers[i].data()+1,buffers[i].size()-2});
                    reader->read(spans);
                    for(std::size_t i=0;i<offsets.size();++i) {
                        require(buffers[i].front()==std::byte{0x5a} && buffers[i].back()==std::byte{0x5a},"MFQ page scatter overwrote guards");
                        require(std::equal(buffers[i].begin()+1,buffers[i].end()-1,expected.begin()+offsets[i]),"MFQ page dedupe/ticket order/EOF differs");
                    }
                }
            }));
            for(auto& job:jobs)job.get();
            const auto stats=reader->stats();require(stats.errors==0 && stats.logical_bytes>stats.physical_bytes,"MFQ did not deduplicate sector reads");
            bool invalid=false;try{reader->read({{expected.size(),expected.data(),1}});}catch(const std::out_of_range&){invalid=true;}
            require(invalid,"MFQ batch accepted a range outside the file");
            std::vector<std::byte> many(400*4,std::byte{0x5a});
            std::vector<mfq::ReadSpan> spans;
            for(std::size_t i=0;i<400;++i)spans.push_back({i*127,many.data()+i*4+1,2});
            reader->read(spans);
            for(std::size_t i=0;i<400;++i) {
                require(many[i*4]==std::byte{0x5a} && many[i*4+3]==std::byte{0x5a},"batched ring overwrote guards");
                require(many[i*4+1]==expected[i*127] && many[i*4+2]==expected[i*127+1],"batched ring wrap changed bytes");
            }
            require(reader->stats().staging_bytes==0 && reader->stats().mode==mode,"batched read mode/staging accounting differs");
            // All ranges are checked before any destination is modified.
            std::byte guard{0x5a};invalid=false;
            try{reader->read({{0,&guard,1},{expected.size(),&guard,1}});}catch(const std::out_of_range&){invalid=true;}
            require(invalid && guard==std::byte{0x5a},"invalid batched range changed an earlier destination");
        }
#if defined(__linux__)
        {
            auto reader=std::make_shared<mfq::FileRangeReader>(path,mfq::FileReadMode::Direct);
            std::vector<std::future<void>> jobs;
            for(int client=0;client<8;++client)jobs.push_back(std::async(std::launch::async,[&,client] {
                for(int round=0;round<12;++round) {
                    std::vector<std::byte> output(96*33,std::byte{0x5a});std::vector<mfq::ReadSpan> spans;
                    for(std::size_t i=0;i<96;++i)spans.push_back({i*167+std::size_t(round+client),output.data()+i*33+1,31});
                    reader->read_batch(spans);
                    for(std::size_t i=0;i<96;++i)
                        require(output[i*33]==std::byte{0x5a} && output[i*33+32]==std::byte{0x5a} &&
                            std::equal(output.begin()+i*33+1,output.begin()+i*33+32,expected.begin()+spans[i].offset),
                            "concurrent Linux batch completion changed bytes");
                }
            }));
            for(auto& job:jobs)job.get();
            require(!reader->stats().staging_bytes && reader->stats().staging_cached_bytes<=(8u<<20),
                "concurrent Linux queues exceeded their staging bound");
            {std::ofstream truncated(path,std::ios::binary|std::ios::trunc);truncated.write("short",5);}
            std::vector<std::byte> output(64,std::byte{0x5a});bool failed=false;
            try{reader->read_batch({{0,output.data(),3},{4096,output.data()+32,31}});}
            catch(const std::runtime_error&){failed=true;}
            require(failed && std::all_of(output.begin(),output.end(),[](auto b){return b==std::byte{0x5a};}) &&
                !reader->stats().staging_bytes,"failed Linux batch published partial data or leaked staging");
            {std::ofstream restored(path,std::ios::binary|std::ios::trunc);
                restored.write(reinterpret_cast<const char*>(expected.data()),expected.size());}
            reader->read_batch({{0,output.data(),3},{4096,output.data()+32,31}});
            require(std::equal(output.begin()+32,output.begin()+63,expected.begin()+4096),
                "Linux queue reused an incomplete failed request");
        }
#endif
        exercise_shared_staging(path);
        std::filesystem::remove(path);
        bool rejected=false;
        try { mfq::FileRangeReader missing(path,mfq::FileReadMode::Direct); } catch (const std::system_error&) { rejected=true; }
        require(rejected,"missing model file was accepted");
        std::cout<<"file range buffered/direct sector edges, unaligned destinations, EOF, unicode and 12-way reads PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::error_code ignored; std::filesystem::remove(path,ignored);
        std::cerr<<error.what()<<'\n'; return 1;
    }
}
