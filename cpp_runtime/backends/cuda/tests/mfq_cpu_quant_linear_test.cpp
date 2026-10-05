#include "storage/weight_loader.h"
#include "quant_linear.h"
#include "cuda_execution.h"

#include <cmath>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::vector<float> read(const std::filesystem::path& path, std::size_t count) {
    std::ifstream stream(path,std::ios::binary);
    std::vector<float> values(count);
    stream.read(reinterpret_cast<char*>(values.data()),count*sizeof(float));
    if (!stream || stream.peek()!=std::char_traits<char>::eof())
        throw std::runtime_error("invalid FP32 fixture: "+path.string());
    return values;
}
}

// Fixture oracle is independent Python canonical pack/unpack and FP64 matmul.
// Only compressed weights enter production CPU dispatch. No dense weight
// expansion is performed by this executable.
int main(int argc, char** argv) try {
    if (argc<2 || argc>3 || (argc==3 && std::string(argv[2])!="--benchmark"))
        throw std::runtime_error("expected canonical CPU quant fixture directory [--benchmark]");
    const bool benchmark=argc==3;
    const std::filesystem::path root(argv[1]);
    std::ifstream manifest(root/"cases.txt");
    if (!manifest) throw std::runtime_error("missing CPU quant fixtures");
    CudaExecutionContext execution;
    execution.loading_cpu_layer=true;
    std::string name;
    int outputs,width,batch,cases=0;
    while (manifest >> name >> outputs >> width >> batch) {
        auto source=mfq::open_model_source((root/(name+".mfq")).string());
        auto layer=load_quant_linear(execution,*source,"linear.weight");
        const auto input=read(root/(name+".input.f32"),static_cast<std::size_t>(batch)*width);
        const auto expected=read(root/(name+".expected.f32"),static_cast<std::size_t>(batch)*outputs);
        auto x=mfq_tensor_backend::tensor(input).reshape({batch,width});
        auto result=layer.forward(execution,x).to(mfq_tensor_backend::kFloat32).contiguous();
        if (result.is_cuda() || result.numel()!=static_cast<int64_t>(expected.size()))
            throw std::runtime_error("CPU quant dispatch has incorrect placement/shape");
        const auto* actual=result.data_ptr<float>();
        double maximum=0, square_error=0, square_reference=0;
        for (std::size_t i=0; i<expected.size(); ++i) {
            const double error=std::abs(double(actual[i])-expected[i]);
            // The public quant-linear interface returns FP16. Allow its
            // rounding plus FP32 accumulation, with an absolute cancellation
            // tolerance fixed before running any fixtures.
            const double tolerance=1e-5+1e-3*std::abs(double(expected[i]));
            if (!std::isfinite(actual[i]) || error>tolerance)
                throw std::runtime_error(name+" differs from canonical FP64 oracle at "+
                    std::to_string(i)+", error="+std::to_string(error));
            maximum=std::max(maximum,error);
            square_error+=error*error;
            square_reference+=double(expected[i])*expected[i];
        }
        std::cout << name << " shape=" << outputs << 'x' << width << " batch=" << batch
            << " max_abs=" << maximum << " relative="
            << std::sqrt(square_error/std::max(square_reference,1e-30)) << '\n';
        if (benchmark) {
            // Separate compressed allocations rotate through 48 layer-shaped
            // weights. This avoids reporting a single L3-resident GEMV.
            std::vector<QuantLinear> layers;
            layers.reserve(48);
            for (int index=0; index<48; ++index)
                layers.push_back(load_quant_linear(execution,*source,"linear.weight"));
            const auto one=mfq_tensor_backend::tensor(
                std::vector<float>(input.begin(),input.begin()+width)).reshape({1,width});
            double checksum=0;
            std::vector<double> times;
            for (int round=0; round<4; ++round) {
                const auto start=std::chrono::steady_clock::now();
                for (auto& item: layers) {
                    const auto y=item.forward(execution,one);
                    checksum+=float(y.data_ptr<mfq_half>()[0]);
                }
                const auto stop=std::chrono::steady_clock::now();
                if (round) times.push_back(std::chrono::duration<double,std::micro>(stop-start).count()/layers.size());
            }
            std::sort(times.begin(),times.end());
            std::cout << "benchmark " << name << " batch=1 layers=48 us_per_projection="
                << times[times.size()/2] << " checksum=" << checksum << std::endl;
        }
        ++cases;
    }
    if (!manifest.eof() || cases==0) throw std::runtime_error("invalid CPU fixture manifest");
    std::cout << "canonical CPU quant-linear cases passed=" << cases << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
