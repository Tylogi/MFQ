#include "packed_nint.h"
#include "quant_dot.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

struct Fixture {
    int outputs, width, group_size, groups;
    std::vector<std::uint8_t> packed, bits, scales, minima;
    std::vector<std::int64_t> offsets;
    std::vector<float> anchors_scale, anchors_min;
    std::vector<std::uint8_t> codes;

    Fixture(int n, int k, int gs, int uniform_bits=0, unsigned seed=41, bool aligned=false)
        : outputs(n), width(k), group_size(gs), groups(1+(k-1)/gs),
          bits(n), scales(n*groups), minima(n*groups), offsets(n),
          anchors_scale(n), anchors_min(n), codes(n*k) {
        std::mt19937 random(seed);
        std::uint64_t position=aligned ? 0 : 3;
        for (int row=0; row<n; ++row) {
            bits[row] = uniform_bits ? uniform_bits : 1+row%8;
            offsets[row] = position;
            position += std::uint64_t(k)*bits[row];
            anchors_scale[row] = (1.0f+float(random()%29))/float(32768*(1u<<bits[row]));
            anchors_min[row] = float(int(random()%23)-11)/8192.0f;
        }
        packed.resize(static_cast<std::size_t>((position+7)/8)); // No implicit padding.
        for (int row=0; row<n; ++row) {
            for (int group=0; group<groups; ++group) {
                scales[row*groups+group] = random()%256;
                minima[row*groups+group] = random()%256;
            }
            for (int column=0; column<k; ++column) {
                const auto value = random()&((1u<<bits[row])-1);
                codes[row*k+column] = value;
                const auto bit = std::uint64_t(offsets[row])+std::uint64_t(column)*bits[row];
                for (int i=0; i<bits[row]; ++i)
                    if ((value>>i)&1u) packed[static_cast<std::size_t>((bit+i)>>3)] |= 1u<<((bit+i)&7);
            }
        }
    }

    mfq::cpu::PackedNintView view() const {
        return {packed.data(),packed.size(),bits.data(),offsets.data(),
            scales.data(),minima.data(),anchors_scale.data(),anchors_min.data(),
            outputs,width,group_size};
    }

    float decoded(int row, int column) const {
        const int meta = row*groups+column/group_size;
        volatile float scale = anchors_scale[row]*scales[meta];
        volatile float minimum = anchors_min[row]*minima[meta];
        volatile float product = scale*codes[row*width+column];
        return product-minimum;
    }
};

void compare(const Fixture& f, int batch, int threads) {
    const int input_stride=f.width+7, output_stride=f.outputs+11;
    std::vector<float> input(batch*input_stride,9876), output(batch*output_stride,-9876), scalar=output;
    std::mt19937 random(91);
    for (int sample=0; sample<batch; ++sample)
        for (int column=0; column<f.width; ++column) {
            float value=float(int(random()%2049)-1024)/128.0f;
            if (sample==1) value *= 1e-19f;
            if (column==f.width/2) value *= 128;
            input[sample*input_stride+column]=value;
        }
    mfq::cpu::packed_nint_matmul(f.view(),input.data(),batch,input_stride,
        output.data(),output_stride,threads);
    mfq::cpu::packed_nint_matmul(f.view(),input.data(),batch,input_stride,
        scalar.data(),output_stride,threads,mfq::cpu::PackedNintKernel::scalar);
    for (int sample=0; sample<batch; ++sample) {
        for (int row=0; row<f.outputs; ++row) {
            double reference=0, magnitude=0;
            for (int column=0; column<f.width; ++column) {
                const double term=double(f.decoded(row,column))*input[sample*input_stride+column];
                reference += term;
                magnitude += std::abs(term);
            }
            // Declared before benchmarking: FP32 arithmetic error <= 2 ppm
            // of the absolute dot-product terms plus 1e-5 absolute error.
            const double tolerance=1e-5+2e-6*magnitude;
            for (const auto* result : {&output,&scalar}) {
                const float value=(*result)[sample*output_stride+row];
                if (!std::isfinite(value) || std::abs(value-reference)>tolerance) {
                    std::cerr << "mismatch shape=" << f.outputs << 'x' << f.width
                        << " row=" << row << " sample=" << sample << " result=" << value
                        << " reference=" << reference << " tolerance=" << tolerance << '\n';
                    throw std::runtime_error("packed NINT differs from FP64 decoded-weight oracle");
                }
            }
        }
        for (int padding=f.outputs; padding<output_stride; ++padding) {
            require(output[sample*output_stride+padding]==-9876,"output stride padding changed");
            require(scalar[sample*output_stride+padding]==-9876,"scalar output padding changed");
        }
    }
    auto zero=input;
    std::fill(zero.begin(),zero.end(),0.0f);
    mfq::cpu::packed_nint_matmul(f.view(),zero.data(),batch,input_stride,
        output.data(),output_stride,threads);
    for (int sample=0; sample<batch; ++sample)
        for (int row=0; row<f.outputs; ++row)
            require(output[sample*output_stride+row]==0,"zero input produced nonzero output");
}

void guarded_tail() {
#ifdef _WIN32
    SYSTEM_INFO info;
    GetSystemInfo(&info);
    const auto page=static_cast<std::size_t>(info.dwPageSize);
    auto* allocation=static_cast<std::uint8_t*>(VirtualAlloc(nullptr,2*page,
        MEM_RESERVE|MEM_COMMIT,PAGE_READWRITE));
    require(allocation!=nullptr,"guard allocation failed");
    DWORD previous;
    require(VirtualProtect(allocation+page,page,PAGE_NOACCESS,&previous)!=0,"guard protection failed");
#else
    const auto page=static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
    auto* allocation=static_cast<std::uint8_t*>(mmap(nullptr,2*page,
        PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0));
    require(allocation!=MAP_FAILED,"guard allocation failed");
    require(mprotect(allocation+page,page,PROT_NONE)==0,"guard protection failed");
#endif
    for (bool aligned : {false,true}) for (int bits=1; bits<=8; ++bits) {
        Fixture f(1,8,8,bits,41,aligned);
        auto view=f.view();
        auto* packed=allocation+page-f.packed.size();
        std::memcpy(packed,f.packed.data(),f.packed.size());
        view.packed=packed;
        float input[8]={1,-2,3,-4,5,-6,7,-8}, result;
        mfq::cpu::packed_nint_matmul(view,input,1,8,&result,1,1);
        double reference=0;
        for (int i=0; i<8; ++i) reference += double(f.decoded(0,i))*input[i];
        require(std::abs(result-reference)<1e-4,"guarded row result differs");
    }
#ifdef _WIN32
    VirtualFree(allocation,0,MEM_RELEASE);
#else
    munmap(allocation,2*page);
#endif
}

void invalid_storage() {
    Fixture f(8,17,7);
    float input[17]={}, output[8];
    auto expect_rejected=[&](mfq::cpu::PackedNintView view) {
        bool rejected=false;
        try { mfq::cpu::packed_nint_matmul(view,input,1,17,output,8,12); }
        catch (const std::invalid_argument&) { rejected=true; }
        require(rejected,"invalid packed row was accepted");
    };
    auto view=f.view();
    view.packed_bytes--;
    expect_rejected(view);
    f.offsets[0]=-1;
    expect_rejected(f.view());
    f.offsets[0]=3;
    f.bits[0]=9;
    expect_rejected(f.view());
    f.bits[0]=1;
    view=f.view();
    view.group_size=0;
    expect_rejected(view);
    view=f.view();
    view.packed=nullptr;
    expect_rejected(view);
}

void signed_dots() {
    std::mt19937 random(119);
    const auto optimized=mfq::cpu::scaled_i8_dot_kernel();
    for (int length=0; length<=64; ++length) {
        std::vector<std::int8_t> weights(length);
        std::vector<float> input(length);
        for (int repeat=0; repeat<128; ++repeat) {
            const float scale=float(int(random()%257)-128)/8192.0f;
            double reference=0, magnitude=0;
            for (int i=0; i<length; ++i) {
                weights[i]=static_cast<std::int8_t>(random()%256-128);
                input[i]=float(int(random()%2049)-1024)/16.0f;
                volatile float decoded=scale*weights[i];
                const double term=double(decoded)*input[i];
                reference+=term;
                magnitude+=std::abs(term);
            }
            for (auto dot : {optimized,mfq::cpu::scaled_i8_dot_scalar})
                require(std::abs(dot(weights.data(),input.data(),length,scale)-reference)
                    <=1e-5+2e-6*magnitude,"signed dot differs from decoded FP64 oracle");
        }
    }
}

void benchmark(int outputs, int width, int bits) {
    // Rotate all 48 layer-shaped expert inputs to avoid measuring just one
    // repeatedly cached matrix. This is an operator benchmark, not token/s.
    constexpr int layers=48, top_k=10;
    std::vector<Fixture> matrices;
    for (int layer=0; layer<layers; ++layer)
        matrices.emplace_back(outputs,width,bits==5 ? 28 : 32,bits,41+layer,true);
    std::vector<float> input(width),output(outputs);
    for (int i=0; i<width; ++i) input[i]=float(i%31-15)/16.0f;
    const int threads=std::max(1u,std::thread::hardware_concurrency());
    auto measure=[&](mfq::cpu::PackedNintKernel kernel) {
        double checksum=0;
        const auto start=std::chrono::steady_clock::now();
        for (int expert=0; expert<top_k; ++expert)
            for (const auto& matrix : matrices) {
                mfq::cpu::packed_nint_matmul(matrix.view(),input.data(),1,width,
                    output.data(),outputs,threads,kernel);
                checksum += output[expert%outputs];
            }
        const auto stop=std::chrono::steady_clock::now();
        return std::make_pair(std::chrono::duration<double,std::milli>(stop-start).count(),checksum);
    };
    const auto scalar=measure(mfq::cpu::PackedNintKernel::scalar);
    const auto vector=measure(mfq::cpu::PackedNintKernel::automatic);
    require(std::abs(scalar.second-vector.second)<0.01,"benchmark checksum differs");
    std::cout << "{\"shape\":[" << outputs << ',' << width << "],\"q_bits\":" << bits
        << ",\"threads\":" << threads << ",\"projections\":" << layers*top_k
        << ",\"scalar_ms\":" << scalar.first << ",\"optimized_ms\":" << vector.first
        << ",\"projection_us\":" << vector.first*1000/(layers*top_k)
        << ",\"speedup\":" << scalar.first/vector.first << "}\n";
}
} // namespace

int main(int argc, char**) {
    try {
        const int threads=std::max(1u,std::thread::hardware_concurrency());
        for (int bits=1; bits<=8; ++bits)
            for (int width : {1,7,8,9,17,63,65})
                for (int gs : {3,7,24,28,32,64})
                    compare(Fixture(3,width,gs,bits),3,threads);
        compare(Fixture(640,2560,32),1,threads);
        compare(Fixture(640,2560,32,4,41,true),1,threads);
        compare(Fixture(640,2560,32,8,41,true),1,threads);
        compare(Fixture(2560,640,28),7,threads);
        guarded_tail();
        invalid_storage();
        signed_dots();
        std::cout << "packed NINT FP32 oracle, mixed rows, strided batches, zero input, bounds: passed; avx2="
            << mfq::cpu::packed_nint_has_avx2() << '\n';
        if (argc>1) {
            benchmark(640,2560,4);
            benchmark(2560,640,5);
            benchmark(640,2560,8);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
