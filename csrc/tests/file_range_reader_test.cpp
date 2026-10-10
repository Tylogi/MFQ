#include "mfq/file_range_reader.h"
#include "mfq/async_range_reader.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace {
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
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
        exercise(path,expected,mfq::FileReadMode::Buffered);
        exercise(path,expected,mfq::FileReadMode::Direct);
        {
            auto reader=std::make_shared<mfq::AsyncRangeReader>(path);
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
        }
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
