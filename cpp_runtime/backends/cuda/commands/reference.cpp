#include "cli.h"
#include "storage/model_loader.h"
#include "cuda_execution.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

namespace mfq::cuda::commands {
namespace {
namespace tb = mfq_tensor_backend;

tb::Tensor rows(const mfq::ModelSource& source, const std::string& name,
                std::int64_t begin, std::int64_t count) {
    const auto* record = source.find_tensor(name);
    if (!record || !record->shape || record->shape->size() != 2)
        throw std::runtime_error("reference requires a matrix: " + name);
    const auto& shape = *record->shape;
    if (begin < 0 || count < 1 || begin > shape[0] - count)
        throw std::runtime_error("reference matrix row range is invalid");
    tb::ScalarType dtype;
    std::size_t bytes;
    if (record->dtype == "F16") { dtype = tb::kFloat16; bytes = 2; }
    else if (record->dtype == "BF16") { dtype = tb::kBFloat16; bytes = 2; }
    else if (record->dtype == "F32") { dtype = tb::kFloat32; bytes = 4; }
    else throw std::runtime_error("reference embedding/output matrix requires a floating source: " + name);
    auto result = tb::empty({count, shape[1]}, tb::TensorOptions().device(tb::kCPU).dtype(dtype));
    source.read_range_into(name, sizeof(std::uint32_t) + 2 * sizeof(std::int64_t) + begin * shape[1] * bytes,
                          reinterpret_cast<std::byte*>(result.data_ptr()), count * shape[1] * bytes);
    return result;
}

template<class Model>
int generate(CudaExecutionContext& execution, const CudaLoadOptions& options,
             const std::filesystem::path& input, const std::filesystem::path& output,
             bool resident) {
    std::ifstream tokens_file(input, std::ios::binary);
    std::uint32_t geometry[2]{};
    tokens_file.read(reinterpret_cast<char*>(geometry), sizeof(geometry));
    const auto chunks = geometry[0], context = geometry[1];
    if (!tokens_file || !chunks || context < 32 || context != options.context_size ||
        std::uint64_t(chunks) * context > std::uint64_t(std::numeric_limits<std::int32_t>::max()))
        throw std::runtime_error("invalid native reference token geometry");
    std::vector<std::int32_t> tokens(std::size_t(chunks) * context);
    tokens_file.read(reinterpret_cast<char*>(tokens.data()), tokens.size() * sizeof(tokens[0]));
    if (!tokens_file || tokens_file.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("invalid native reference token payload");
    if (std::filesystem::exists(output)) throw std::runtime_error("reference scratch output already exists");
    auto model = load_causal_lm<Model>(execution, options.model_path, options.config_path,
                                      context, false, false, {}, false);
    const auto& source = *model.source;
    const auto width = model.hidden_size();
    const auto vocab = model.vocab_size();
    for (auto token : tokens) if (token < 0 || token >= vocab)
        throw std::runtime_error("reference token is outside the source vocabulary");
    const auto hidden_path = output.string() + ".hidden";
    std::fstream hidden(hidden_path, std::ios::in | std::ios::out | std::ios::binary | std::ios::trunc);
    if (!hidden) throw std::runtime_error("cannot create reference hidden-state scratch");
    const auto hidden_bytes = std::uint64_t(context) * width * sizeof(float);
    std::vector<float> buffer(std::size_t(context) * width);
    auto store = [&](std::uint32_t chunk, tb::Tensor value) {
        auto cpu = value.to(tb::kCPU, tb::kFloat32).contiguous();
        hidden.seekp(chunk * hidden_bytes);
        hidden.write(reinterpret_cast<const char*>(cpu.template data_ptr<float>()), hidden_bytes);
        hidden.flush();
        if (!hidden) throw std::runtime_error("cannot write reference hidden-state scratch");
    };
    auto restore = [&](std::uint32_t chunk) {
        hidden.seekg(chunk * hidden_bytes);
        hidden.read(reinterpret_cast<char*>(buffer.data()), hidden_bytes);
        if (!hidden) throw std::runtime_error("cannot read reference hidden-state scratch");
        return tb::from_blob(buffer.data(), {1, context, width}, tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32))
            .to(tb::kCUDA, tb::kFloat16).contiguous();
    };
    for (std::uint32_t chunk = 0; chunk < chunks; ++chunk) {
        auto embedded = tb::empty({1, context, width}, tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat16));
        for (std::uint32_t position = 0; position < context; ++position) {
            auto row = rows(source, "model.token_embedding.weight", tokens[std::size_t(chunk) * context + position], 1)
                           .to(tb::kFloat16);
            embedded.narrow(1, position, 1).copy_(row.reshape({1, 1, width}));
        }
        store(chunk, model.adapter_prepare_hidden(embedded.to(tb::kCUDA), 1, context));
    }
    auto positions = tb::arange(0, context, tb::TensorOptions().device(tb::kCUDA).dtype(tb::kInt64));
    if (resident) {
        try {
            for (int index = 0; index < model.num_hidden_layers(); ++index)
                model.blocks.push_back(model.adapter_load_block(source, index, 0, std::string(model.layer_type(index))));
        } catch (const std::exception& error) {
            const std::string message = error.what();
            if (message.find("out of memory") == std::string::npos &&
                message.find("memory allocation") == std::string::npos) throw;
            model.blocks.clear();
            mfq_cuda_empty_cache();
            resident = false;
            std::cout << "{\"event\":\"source_loading\",\"layerwise\":true,\"requested_layerwise\":false,"
                         "\"fallback_reason\":\"insufficient_memory\"}" << std::endl;
        }
    }
    for (int index = 0; index < model.num_hidden_layers(); ++index) {
        auto streamed = resident ? std::unique_ptr<Block>{} :
            model.adapter_load_block(source, index, 0, std::string(model.layer_type(index)));
        auto& block = resident ? model.blocks[index] : streamed;
        for (std::uint32_t chunk = 0; chunk < chunks; ++chunk) {
            block->reset(1);
            auto value = block->forward(execution, restore(chunk), positions, 0, mfq_nullopt, model.rope);
            store(chunk, value);
        }
        mfq_cuda_synchronize();
        streamed.reset();
        if (!resident) mfq_cuda_empty_cache();
        std::cout << "{\"event\":\"wt2_layer\",\"layer\":" << index + 1
                  << ",\"layers\":" << model.num_hidden_layers() << "}" << std::endl;
    }
    model.blocks.clear();
    mfq_cuda_empty_cache();
    const auto head = !model.tie_word_embeddings() && source.find_tensor("model.output.weight")
        ? "model.output.weight" : "model.token_embedding.weight";
    std::ofstream logits(output, std::ios::binary);
    if (!logits) throw std::runtime_error("cannot create reference logits scratch");
    const std::int64_t first = context / 2, scored = context - first - 1;
    for (std::uint32_t chunk = 0; chunk < chunks; ++chunk) {
        auto normalized = model.finalize_hidden(restore(chunk), 1, context);
        for (std::int64_t begin = 0; begin < scored; begin += 32) {
            const auto count = std::min<std::int64_t>(32, scored - begin);
            auto value = normalized.narrow(1, first + begin, count).reshape({count, width}).to(tb::kFloat16);
            std::vector<float> row_logits(count * vocab);
            for (std::int64_t row = 0; row < vocab; row += 4096) {
                const auto rows_count = std::min<std::int64_t>(4096, vocab - row);
                auto weight = rows(source, head, row, rows_count).to(tb::kCUDA, tb::kFloat16);
                auto result = model.apply_final_logit_softcap(tb::matmul(value, weight.transpose(0, 1)))
                    .to(tb::kCPU, tb::kFloat32).contiguous();
                for (std::int64_t token = 0; token < count; ++token)
                    std::memcpy(row_logits.data() + token * vocab + row,
                                result.template data_ptr<float>() + token * rows_count, rows_count * sizeof(float));
            }
            logits.write(reinterpret_cast<const char*>(row_logits.data()), row_logits.size() * sizeof(float));
            if (!logits) throw std::runtime_error("cannot write reference logits scratch");
        }
        std::cout << "{\"event\":\"wt2_head\",\"chunk\":" << chunk + 1
                  << ",\"chunks\":" << chunks << "}" << std::endl;
    }
    logits.close();
    if (!logits) throw std::runtime_error("cannot finalize reference logits scratch");
    hidden.close();
    std::filesystem::remove(hidden_path);
    return 0;
}
}

int run_reference(int argc, char** argv) {
    return internal::with_command_errors([&] {
        ArgCursor args(argc, argv, 2);
        CudaLoadOptions options;
        std::filesystem::path input, output;
        bool resident = false;
        while (!args.empty()) {
            const auto option = args.next();
            if (parse_cuda_load_option(option, args, options)) continue;
            if (option == "--reference-input") input = args.value(option);
            else if (option == "--reference-output") output = args.value(option);
            else if (option == "--resident") resident = true;
            else usage_error("unknown native reference option: " + std::string(option));
        }
        if (options.model_path.empty() || input.empty() || output.empty() || options.context_size < 32)
            usage_error("native reference requires --model, --ctx-size, --reference-input and --reference-output");
        if (!std::filesystem::is_directory(options.model_path))
            usage_error("native reference requires an original HF model directory");
        CudaExecutionContext execution;
        internal::setup_cuda_load(options, execution);
        tb::NoGradGuard guard;
        auto source = mfq::open_model_source(options.model_path);
        const auto backbone = source->resolved_model_graph().backbone;
        if (backbone == "qwen3_5") return generate<Qwen35CausalLm>(execution, options, input, output, resident);
        if (backbone == "gemma4") return generate<Gemma4CausalLm>(execution, options, input, output, resident);
        throw std::runtime_error("native reference architecture is unsupported: " + backbone);
    });
}
} // namespace mfq::cuda::commands
