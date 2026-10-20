#include "models/deepseek_v41/ops.h"
#include "native/context.h"
#include "csrc/tests/quant/nint_row_fixture.h"
#include <atomic>
#include <iostream>

namespace {
namespace tb = mfq_tensor_backend;
using namespace mfq::cuda::deepseek_v41_runtime;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

class Rows final : public mfq::ModelSource {
  public:
    static constexpr int count = 11, width = 64;
    bool batched;
    std::vector<uint8_t> data;
    mutable std::atomic<int> reads{0};
    mutable std::atomic<int> mapped_reads{0};
    bool mapped=false;
    std::vector<float> nint_reference;
    std::vector<mfq::TensorMetadata> records;
    Rows(bool batch,bool nint=false,bool mapping=false) : batched(batch), data(56 + count * (width + width / 32), 0),mapped(mapping) {
        if(nint) {
            auto fixture=mfq::test::fixture(count,width,24,6);
            data=std::move(fixture.blob);nint_reference=std::move(fixture.reference);
            records.push_back({"table","NINT","NINT",{},data.size()});
            return;
        }
        std::memcpy(data.data(), "MXT1", 4); data[4] = 1; data[5] = 8;
        const uint64_t header[]{count, width, count, width, count, width / 32};
        std::memcpy(data.data() + 8, header, sizeof(header));
        for (int row = 0; row < count; ++row) {
            for (int col = 0; col < width; ++col) {
                auto value = uint8_t((row * 13 + col * 7) % 127);
                data[56 + row * width + col] = value | ((col % 3 == 0) ? 128 : 0);
            }
            for (int group = 0; group < width / 32; ++group)
                data[56 + count * width + row * (width / 32) + group] = uint8_t(124 + (row + group) % 5);
        }
        records.push_back({"table", "MXFP8", "MXFP8", {}, data.size()});
    }
    const std::vector<std::filesystem::path>& source_paths() const noexcept override {
        static const std::vector<std::filesystem::path> paths; return paths;
    }
    std::string_view architecture() const noexcept override { return "deepseek_v41"; }
    const std::unordered_map<std::string,std::string>& metadata() const noexcept override {
        static const std::unordered_map<std::string,std::string> metadata; return metadata;
    }
    const std::vector<mfq::TensorMetadata>& tensors() const noexcept override { return records; }
    const mfq::TensorMetadata* find_tensor(std::string_view name) const noexcept override {
        return name == "table" ? &records.front() : nullptr;
    }
    void read_range_into(std::string_view name, uint64_t offset, std::byte* dest, size_t size) const override {
        require(name == "table" && offset <= data.size() && size <= data.size() - offset,
                "fixture read out of bounds");
        ++reads;
        std::memcpy(dest, data.data() + offset, size);
    }
    TensorBatchReader tensor_batch_reader(std::string_view) const override {
        if (!batched) return {};
        return [this](const std::vector<mfq::ReadSpan>& spans) {
            for (auto span : spans) read_range_into("table",span.offset,span.destination,span.size);
        };
    }
    TensorReader tensor_reader(std::string_view) const override {
        return [this](uint64_t offset,std::byte* destination,std::size_t bytes) {
            read_range_into("table",offset,destination,bytes);
        };
    }
    TensorBatchReader mapped_tensor_batch_reader(std::string_view) const override {
        if(!mapped)return {};
        return [this](const std::vector<mfq::ReadSpan>& spans) {
            ++mapped_reads;
            for(auto span:spans)read_range_into("table",span.offset,span.destination,span.size);
        };
    }
    bool supports_parallel_tensor_reads() const noexcept override { return !batched; }
    const std::vector<std::string>& assets() const noexcept override {
        static const std::vector<std::string> names; return names;
    }
    bool has_asset(std::string_view) const noexcept override { return false; }
    std::vector<std::byte> read_asset(std::string_view) const override { return {}; }
    std::optional<mfq::ModelGraph> model_graph() const override { return {}; }
    float decoded(int row, int col) const {
        if(!nint_reference.empty())return float(mfq_half(nint_reference[row*width+col]));
        const auto value = engram_detail::decode_e4m3(data[56 + row * width + col]);
        const int scale = data[56 + count * width + row * (width / 32) + col / 32];
        return value * std::ldexp(1.f, scale - 127);
    }
};

void check_rows(bool batched, size_t capacity,bool nint=false,bool mapped=false) {
    Rows source(batched,nint,mapped);
    auto table = EngramTable::load(source,"table", Rows::count, Rows::width, capacity);
    const auto device = tb::Device(tb::kCUDA, 0);
    const auto check = [&](const std::vector<int64_t>& ids, int batch, int tokens, int columns) {
        table->prefetch(ids.data(), ids.size());
        auto result = table->gather(ids.data(),batch,tokens,columns,device).to(tb::kFloat32).cpu();
        for (size_t row = 0; row < ids.size(); ++row) for (int col = 0; col < Rows::width; ++col)
            require(result.data_ptr<float>()[row * Rows::width + col] == source.decoded(int(ids[row]),col),
                    "Engram packed gather changed values/order");
    };
    check({7,2,7,1,2,1},1,2,3);
    const int reads = source.reads;
    check({7,2,7,1,2,1},1,2,3);
    require(source.reads == reads,"identical Engram row request repeated SSD reads");
    check({9,0,3,8,10,3,4,2,9,8,6,5},2,2,3);
    check({1,7,1},1,1,3);
    int64_t invalid = Rows::count;
    bool rejected = false;
    try { table->prefetch(&invalid,1); } catch(const std::out_of_range&) { rejected = true; }
    require(rejected,"Engram accepted an out of bounds row");
    // A discarded asynchronous request must complete before cache reuse.
    const int64_t first[]{8,9}, second[]{4,6};
    table->prefetch(first,2); table->prefetch(second,2);
    check({4,6},1,1,2);
    require((source.mapped_reads>0)==mapped,"Engram row source did not use the selected reader");
}

void check_bad_row(bool scale) {
    Rows source(true);
    source.data[scale ? 56 + Rows::count * Rows::width : 56] = scale ? 255 : 127;
    auto table = EngramTable::load(source,"table",Rows::count,Rows::width,2);
    int64_t row = 0;
    bool rejected = false;
    try { table->gather(&row,1,1,1,tb::Device(tb::kCUDA,0)); }
    catch(const std::runtime_error&) { rejected = true; }
    require(rejected,"Engram accepted a non-finite encoded row");
}

tb::Tensor reference_gate(const tb::Tensor& hidden,const tb::Tensor& projected,
                          const tb::Tensor& weight,double eps) {
    const auto batch=hidden.size(0),tokens=hidden.size(1),streams=hidden.size(2),width=hidden.size(3);
    auto key=projected.narrow(-1,0,streams*width).reshape({batch,tokens,streams,width}).to(tb::kFloat32);
    auto value=projected.narrow(-1,streams*width,width).reshape({batch,tokens,1,width}).to(tb::kFloat32);
    auto source=hidden.to(tb::kFloat32);
    auto inverse=tb::rsqrt(source.square().mean(-1)+eps)*tb::rsqrt(key.square().mean(-1)+eps);
    auto score=(source*key*weight).sum(-1)*inverse/std::sqrt(double(width));
    auto sign=(score>0.).to(tb::kFloat32)-(score<0.).to(tb::kFloat32);
    auto gate=tb::sigmoid(sign*score.abs().clamp_min(1.e-6).sqrt());
    return (source+gate.unsqueeze(-1)*value).to(tb::kFloat16).contiguous();
}

void check_gate(int batch,int tokens,int width,bool fp32) {
    auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat32);
    auto hidden=tb::randn({batch,tokens,4,width},options).to(tb::kFloat16).contiguous();
    auto projected=tb::randn({batch,tokens,5*width},options).to(fp32?tb::kFloat32:tb::kFloat16).contiguous();
    auto weight=tb::randn({4,width},options);
    auto expected=reference_gate(hidden,projected,weight,1.e-20);
    auto actual=deepseek_v41_engram_gate_add_cuda(hidden,projected,weight,1.e-20);
    const auto error=(actual.to(tb::kFloat32)-expected.to(tb::kFloat32)).abs().max().item<double>();
    require(std::isfinite(error) && error<=.00391,"Engram fused gate/add changed output");
    hidden.zero_(); projected.zero_();
    require(deepseek_v41_engram_gate_add_cuda(hidden,projected,weight,1.e-20).equal(hidden),
            "Engram zero gate is not finite");
}

template<class F> double elapsed(F function,int repetitions) {
    for(int i=0;i<8;++i)function();
    mfq::cuda::Event begin(cudaEventDefault),end(cudaEventDefault);
    auto stream=mfq_current_cuda_stream();
    begin.record(stream);
    for(int i=0;i<repetitions;++i)function();
    end.record(stream);end.synchronize();
    float ms=0;MFQ_CUDA_CHECK(cudaEventElapsedTime(&ms,begin.get(),end.get()));
    return ms/repetitions;
}
void benchmark_gate() {
    auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat32);
    for(int tokens:{1,6,128,1024,4096}) {
        auto hidden=tb::randn({1,tokens,4,5120},options).to(tb::kFloat16).contiguous();
        auto projected=tb::randn({1,tokens,5*5120},options).to(tb::kFloat16).contiguous();
        auto weight=tb::randn({4,5120},options);
        auto previous=[&]{return reference_gate(hidden,projected,weight,1.e-20);};
        auto fused=[&]{return deepseek_v41_engram_gate_add_cuda(hidden,projected,weight,1.e-20);};
        const auto error=(previous().to(tb::kFloat32)-fused().to(tb::kFloat32)).abs().max().item<double>();
        require(std::isfinite(error) && error<=.00391,"Engram benchmark output mismatch");
        const int repetitions=tokens<=6?200:tokens<=128?80:tokens<=1024?15:5;
        const auto old_ms=elapsed(previous,repetitions),new_ms=elapsed(fused,repetitions);
        std::cout<<"{\"tokens\":"<<tokens<<",\"previous_ms\":"<<old_ms
                 <<",\"fused_ms\":"<<new_ms<<",\"speedup\":"<<old_ms/new_ms<<"}"<<std::endl;
    }
}
}

int main(int argc,char** argv) try {
    int devices=0;
    if(cudaGetDeviceCount(&devices)!=cudaSuccess || !devices)return 77;
    if(argc==2 && std::string(argv[1])=="--bench") {benchmark_gate();return 0;}
    check_rows(true,64); check_rows(false,2); check_rows(false,0);
    check_rows(true,64,true);check_rows(false,2,true);check_rows(false,0,true);
    check_rows(true,64,true,true);check_rows(false,2,true,true);check_rows(false,0,true,true);
    check_bad_row(false); check_bad_row(true);
    check_gate(1,1,5120,false); check_gate(2,3,67,true); check_gate(1,128,5120,false);
    std::cout<<"DeepSeek-V4.1 Engram row pipeline and fused gate tests passed\n";
    return 0;
} catch(const std::exception& error) {
    std::cerr<<error.what()<<'\n';return 1;
}
