#include "storage/model_loader.h"
#include "cuda_runtime_config.h"
#include <cuda_runtime_api.h>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace tb = mfq_tensor_backend;
using Json = nlohmann::json;

static Json descriptor(const tb::Tensor& value) {
    auto host = value.to(tb::kFloat32).cpu().contiguous();
    return {{"shape", host.sizes().vec()},
            {"data", std::vector<float>(host.data_ptr<float>(), host.data_ptr<float>() + host.numel())}};
}

int main(int argc, char** argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "SKIP: no CUDA device\n";
        return 77;
    }
    if (argc == 1 || (argc == 2 && std::string(argv[1]) == "--probe")) return 0;
    try {
        if (argc != 3) throw std::invalid_argument("usage: mfq-rms-swiglu-vision-test MODEL INPUT.json");
        const std::string input_path = argv[2];
        std::ifstream input{std::filesystem::path(std::u8string(input_path.begin(), input_path.end()))};
        Json request;
        input >> request;
        mfq::cuda::CudaLoadOptions options;
        options.model_path = argv[1]; options.context_size = 128;
        return mfq::cuda::internal::with_cuda_load(options, [&](CudaExecutionContext& execution) {
            auto model = mfq::cuda::load_causal_lm<mfq::cuda::Glm5CausalLm>(execution,
                options.model_path, "", options.context_size, true, true);
            auto components = load_runtime_components(model, true);
            if (!components.rms_vision) throw std::runtime_error("missing RMS-SwiGLU vision adapter");
            MfqMultimodalInput media;
            media.processor = MfqMultimodalProcessor::grid_vision;
            media.processor_name = request.value("processor", std::string(mfq::kMfqGridVisionInputContract));
            media.pixel_values = request.at("pixels").get<std::vector<float>>();
            media.pixel_shape = request.at("pixel_shape").get<std::vector<int64_t>>();
            const auto grids = request.at("grids").get<std::vector<std::vector<int32_t>>>();
            media.vision_types = request.at("types").get<std::vector<int32_t>>();
            for (size_t i = 0; i < grids.size(); ++i) {
                const auto& grid = grids.at(i);
                media.vision_grid.insert(media.vision_grid.end(), grid.begin(), grid.end());
                auto& split = media.vision_types.at(i) == 1 ? media.image_grid : media.video_grid;
                split.insert(split.end(), grid.begin(), grid.end());
            }
            media.vision_grid_shape = {static_cast<int64_t>(grids.size()), 3};
            media.image_grid_shape = {static_cast<int64_t>(media.image_grid.size()/3), 3};
            media.video_grid_shape = {static_cast<int64_t>(media.video_grid.size()/3), 3};
            const auto ids = request.at("ids").get<std::vector<int64_t>>();
            Json result;
            // A repeated preparation also exercises the cache without relying
            // on generated text to observe the injected embeddings.
            for (int repeat = 0; repeat < 2; ++repeat) {
                auto preparation = components.rms_vision->prepare(model, ids, media);
                CudaPreparedPrompt prepared;
                while (auto step = preparation.next())
                    if (step.value) prepared = std::move(*step.value);
                if (!prepared.embeddings.defined()) throw std::runtime_error("vision preparation returned no embeddings");
                result[repeat ? "cached_embeddings" : "embeddings"] = descriptor(prepared.embeddings);
                if (!repeat) {
                    result["positions"] = descriptor(prepared.positions);
                    auto tokens = tb::tensor(ids, tb::TensorOptions().device(tb::kCUDA).dtype(tb::kInt64)).reshape({1,-1});
                    model.reset(1);
                    result["logits"] = descriptor(model.forward_inputs(tokens, prepared.embeddings, prepared.positions));
                }
            }
            result["vision_available"] = components.vision_available;
            result["mtp_available"] = components.mtp_available;
            std::cout << result.dump() << '\n';
            return 0;
        });
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
