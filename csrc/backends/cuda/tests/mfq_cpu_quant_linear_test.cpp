#include "storage/weight_loader.h"
#include "quant_linear.h"
#include "cuda_execution.h"
#include "mfq/nint_rows.h"
#include "mfq/nvq_rows.h"

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

void verify_slices(CudaExecutionContext& execution,const mfq::ModelSource& source,
        const std::string& name,const mfq_tensor_backend::Tensor& input,
        const std::vector<float>& expected,int outputs,int batch) {
    const auto& record=require_tensor(source,"linear.weight");
    const auto reader=source.tensor_reader("linear.weight");
    std::size_t reads=0;
    const auto read=[&](std::size_t offset,std::uint8_t* data,std::size_t bytes) {
        reads+=bytes; reader(offset,reinterpret_cast<std::byte*>(data),bytes);
    };
    std::unique_ptr<mfq::NintRows> nint;
    std::unique_ptr<mfq::NvqRows> nvq;
    if (record.dtype=="NINT") nint=std::make_unique<mfq::NintRows>(record.nbytes,read);
    else nvq=std::make_unique<mfq::NvqRows>(record.nbytes,read);
    if (reads>=record.nbytes) throw std::runtime_error("range index read entire weight payload");
    for (const auto interval: {std::pair<int,int>{1,17},{outputs/2,outputs/2+9},{outputs-11,outputs},{0,outputs}}) {
        const int begin=interval.first,end=interval.second,count=end-begin;
        const auto blob=nint ? nint->slice_rows_blob(begin,end) : nvq->slice_rows_blob(begin,end);
        if (blob.size()!=(nint ? nint->row_range_nbytes(begin,end) : nvq->row_range_nbytes(begin,end)))
            throw std::runtime_error("canonical row range byte budget mismatch");
        if (!begin && end==outputs) {
            std::vector<std::uint8_t> original(record.nbytes);
            reader(0,reinterpret_cast<std::byte*>(original.data()),original.size());
            if (blob!=original) throw std::runtime_error("range slice changed canonical whole-weight bytes");
        }
        QuantLinear sliced;
        sliced.logical_out=count; sliced.logical_neuron_len=input.size(1);
        if (nint) { sliced.kind=QuantLinearKind::Nint; sliced.nint=to_cpu_nint(unpack_nint(blob)); }
        else { sliced.kind=QuantLinearKind::Nvq; sliced.nvq=to_cpu_nvq(unpack_nvq(blob,record.dtype)); }
        const auto y=sliced.forward(execution,input).to(mfq_tensor_backend::kFloat32).contiguous();
        if (y.is_cuda() || y.size(0)!=batch || y.size(1)!=count)
            throw std::runtime_error("range slice output placement/shape mismatch");
        for (int sample=0; sample<batch; ++sample)
            for (int row=0; row<count; ++row) {
                const double reference=expected[sample*outputs+begin+row];
                const double actual=y.data_ptr<float>()[sample*count+row];
                if (!std::isfinite(actual) || std::abs(actual-reference)>1e-5+1e-3*std::abs(reference))
                    throw std::runtime_error(name+" range slice differs from canonical FP64 oracle");
            }
    }
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
        verify_slices(execution,*source,name,x,expected,outputs,batch);
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
