#include "causal_lm.h"

int run_minicpmo45_duplex(
        const std::string & model_path,
        const std::string & config_path,
        const std::string & input_prefix,
        const std::string & output_prefix,
        int64_t context_size,
        int64_t steps,
        int64_t max_speak_tokens,
        bool greedy,
        int64_t seed) {
    mfq_tensor_backend::NoGradGuard no_grad;
    g_profiler.enabled = false;
    if (input_prefix.empty() || output_prefix.empty() || steps <= 0 ||
            max_speak_tokens < 2 || seed < 0) {
        throw std::runtime_error(
            "MiniCPM-o duplex mode requires prefixes, positive steps, "
            "at least two generation slots, and a non-negative seed");
    }
    mfq_tensor_backend::manual_seed(seed);
    mfq_cuda_manual_seed_all(seed);
    auto runtime = MiniCPMO45Runtime::load(
        model_path, config_path, context_size);
    auto special_ids = MiniCPMO45DuplexSpecialIds::from_tensor(
        minicpmo45_load_tensor(input_prefix + ".special_ids.pt", true));
    std::vector<int64_t> forbidden_ids;
    auto forbidden = minicpmo45_load_tensor(
        input_prefix + ".forbidden_ids.pt", false);
    if (forbidden.defined()) {
        forbidden = forbidden.to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt64)
            .contiguous().reshape({-1});
        const auto * values = forbidden.data_ptr<int64_t>();
        forbidden_ids.assign(values, values + forbidden.numel());
    }
    MiniCPMO45DuplexSession session(
        runtime, special_ids, std::move(forbidden_ids), greedy);
    const auto system_ids = minicpmo45_load_tensor(
        input_prefix + ".system_ids.pt", false);
    session.prepare(system_ids);

    int64_t listen_steps = 0;
    int64_t speak_steps = 0;
    for (int64_t step = 0; step < steps; ++step) {
        const std::string input = minicpmo45_duplex_step_prefix(
            input_prefix, step);
        const std::string output = minicpmo45_duplex_step_prefix(
            output_prefix, step);
        const bool reset_session = minicpmo45_load_optional_scalar(
            input + ".reset_session.pt", 0) != 0;
        if (reset_session) session.prepare(system_ids);
        const auto pixels = minicpmo45_load_tensor(
            input + ".pixel_values.pt", false);
        const auto patch_mask = minicpmo45_load_tensor(
            input + ".patch_mask.pt", false);
        const auto target_sizes = minicpmo45_load_tensor(
            input + ".target_sizes.pt", false);
        const auto image_slice_counts = minicpmo45_load_tensor(
            input + ".image_slice_counts.pt", false);
        const auto audio_features = minicpmo45_load_tensor(
            input + ".audio_features.pt", false);
        const auto text_ids = minicpmo45_load_tensor(
            input + ".text_ids.pt", false);
        const int64_t prefix_extra = minicpmo45_load_optional_scalar(
            input + ".audio_prefix_extra_frames.pt",
            session.audio_chunk_index == 0 ? 0 : 2);
        const int64_t suffix_extra = minicpmo45_load_optional_scalar(
            input + ".audio_suffix_extra_frames.pt", 2);
        const bool force_listen = minicpmo45_load_optional_scalar(
            input + ".force_listen.pt", 0) != 0;

        auto result = session.run_step(
            pixels, patch_mask, target_sizes, image_slice_counts,
            audio_features, prefix_extra, suffix_extra, text_ids,
            max_speak_tokens, force_listen);
        minicpmo45_save_tensor(
            result.decision_logits, output + ".decision_logits.pt");
        minicpmo45_save_tensor(
            result.audio_embeddings, output + ".audio_embeddings.pt");
        minicpmo45_save_tensor(
            result.generated_ids, output + ".generated_ids.pt");
        minicpmo45_save_tensor(
            result.tts_codes, output + ".tts_codes.pt");
        minicpmo45_save_tensor(
            mfq_tensor_backend::tensor(
                std::vector<int64_t>{
                    result.is_listen ? 1 : 0,
                    result.end_of_turn ? 1 : 0,
                    runtime.language.cache_pos,
                    runtime.audio.cache_length(),
                    runtime.tts.cache_position,
                    session.audio_chunk_index},
                mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64)),
            output + ".state.pt");
        if (result.is_listen) {
            ++listen_steps;
        } else {
            ++speak_steps;
        }
    }
    std::cout << "minicpmo45_duplex=ok"
              << " steps=" << steps
              << " listens=" << listen_steps
              << " speaks=" << speak_steps
              << " language_cache=" << runtime.language.cache_pos
              << " audio_cache=" << runtime.audio.cache_length()
              << " tts_cache=" << runtime.tts.cache_position
              << " seed=" << seed << "\n";
    return 0;
}


static mfq_tensor_backend::Tensor minicpmo45_eval_embeddings(
        MiniCPMO45Runtime & runtime,
        mfq_tensor_backend::Tensor input_ids,
        const mfq_tensor_backend::Tensor & image_embeddings,
        mfq_tensor_backend::Tensor image_bounds,
        const mfq_tensor_backend::Tensor & audio_embeddings = mfq_tensor_backend::Tensor(),
        mfq_tensor_backend::Tensor audio_bounds = mfq_tensor_backend::Tensor(),
        mfq_tensor_backend::Tensor audio_valid_lengths = mfq_tensor_backend::Tensor()) {
    if (input_ids.dim() == 1) input_ids = input_ids.unsqueeze(0);
    if (input_ids.dim() != 2 || input_ids.size(0) != 1) {
        throw std::runtime_error(
            "MiniCPM-o eval batch input_ids must have shape [1,tokens]");
    }
    input_ids = input_ids.to(mfq_tensor_backend::kCUDA, mfq_tensor_backend::kInt64).contiguous();
    auto embeddings = runtime.language.embed_forward(input_ids);
    const auto bounds = minicpmo45_parse_bounds(image_bounds, "image");
    const int64_t image_count = image_embeddings.defined()
        ? image_embeddings.size(0) : 0;
    if (bounds.size() != static_cast<size_t>(image_count)) {
        throw std::runtime_error(
            "MiniCPM-o eval batch requires one image bound per frame");
    }
    for (const auto & bound : bounds) {
        if (bound.batch != 0 || bound.source >= image_embeddings.size(0) ||
                bound.end > input_ids.size(1) ||
                bound.end - bound.begin != image_embeddings.size(1)) {
            throw std::runtime_error(
                "MiniCPM-o eval image bound does not match resampler output");
        }
        embeddings.index({
            0, Slice(bound.begin, bound.end), Slice()})
            .copy_(image_embeddings.index({bound.source})
                .to(embeddings.scalar_type()));
    }
    const auto audios = minicpmo45_parse_bounds(audio_bounds, "audio");
    if (!audios.empty()) {
        if (!audio_embeddings.defined() || !audio_valid_lengths.defined()) {
            throw std::runtime_error(
                "MiniCPM-o eval batch audio bounds require audio embeddings");
        }
        audio_valid_lengths = audio_valid_lengths.to(
            mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt64).contiguous().reshape({-1});
        if (audio_valid_lengths.numel() != audio_embeddings.size(0)) {
            throw std::runtime_error(
                "MiniCPM-o eval batch audio length count is invalid");
        }
        const auto * valid_lengths = audio_valid_lengths.data_ptr<int64_t>();
        std::vector<bool> used_audio(
            static_cast<size_t>(audio_embeddings.size(0)), false);
        for (const auto & bound : audios) {
            const int64_t required = bound.end - bound.begin;
            if (bound.batch != 0 || bound.source < 0 ||
                    bound.source >= audio_embeddings.size(0) ||
                    bound.end > input_ids.size(1)) {
                throw std::runtime_error(
                    "MiniCPM-o eval audio bound does not match Whisper output");
            }
            const int64_t available = valid_lengths[bound.source];
            if (available < 0 || available > audio_embeddings.size(1) ||
                    required != available) {
                throw std::runtime_error(
                    "MiniCPM-o eval audio bound length does not match its source");
            }
            embeddings.index({
                0, Slice(bound.begin, bound.end), Slice()})
                .copy_(audio_embeddings.index({
                    bound.source, Slice(0, available), Slice()})
                    .to(embeddings.scalar_type()));
            used_audio[static_cast<size_t>(bound.source)] = true;
        }
        if (std::find(used_audio.begin(), used_audio.end(), false) !=
                used_audio.end()) {
            throw std::runtime_error(
                "MiniCPM-o eval batch has unused Whisper segments");
        }
    }
    return embeddings;
}

static bool minicpmo45_contains_token(
        const std::vector<int64_t> & values,
        int64_t token) {
    return std::find(values.begin(), values.end(), token) != values.end();
}

static std::vector<std::pair<int64_t, int64_t>>
minicpmo45_prefill_segments(
        mfq_tensor_backend::Tensor boundaries,
        int64_t range_begin,
        int64_t range_end,
        int64_t chunk_tokens) {
    if (range_begin < 0 || range_end <= range_begin || chunk_tokens < 0) {
        throw std::runtime_error(
            "MiniCPM-o prefill segment range is invalid");
    }
    boundaries = boundaries.to(
        mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt64).contiguous().reshape({-1});
    if (boundaries.numel() == 0) {
        throw std::runtime_error(
            "MiniCPM-o segmented prefill requires boundaries");
    }
    const auto * values = boundaries.data_ptr<int64_t>();
    std::vector<int64_t> eligible;
    eligible.reserve(static_cast<size_t>(boundaries.numel()));
    bool found_begin = range_begin == 0;
    bool found_end = false;
    int64_t previous = 0;
    for (int64_t index = 0; index < boundaries.numel(); ++index) {
        const int64_t boundary = values[index];
        if (boundary <= previous) {
            throw std::runtime_error(
                "MiniCPM-o prefill boundaries must be increasing");
        }
        previous = boundary;
        if (boundary == range_begin) found_begin = true;
        if (boundary > range_begin && boundary <= range_end) {
            eligible.push_back(boundary);
        }
        if (boundary == range_end) found_end = true;
    }
    if (!found_begin || !found_end) {
        throw std::runtime_error(
            "MiniCPM-o segmented prefill omits a prompt span");
    }

    std::vector<std::pair<int64_t, int64_t>> segments;
    int64_t segment_begin = range_begin;
    for (const int64_t boundary : eligible) {
        if (chunk_tokens == 0 ||
                boundary == range_end ||
                boundary - segment_begin >= chunk_tokens) {
            segments.emplace_back(segment_begin, boundary);
            segment_begin = boundary;
        }
    }
    if (segments.empty() || segments.back().second != range_end) {
        throw std::runtime_error(
            "MiniCPM-o segmented prefill omits a prompt span");
    }
    return segments;
}

static std::vector<std::pair<int64_t, int64_t>>
minicpmo45_teacher_prefill_segments(
        mfq_tensor_backend::Tensor boundaries,
        int64_t range_end,
        int64_t text_begin,
        int64_t text_end,
        int64_t chunk_tokens) {
    if (range_end <= 0 || text_begin < 0 || text_end <= text_begin ||
            text_end > range_end || chunk_tokens < 0) {
        throw std::runtime_error(
            "MiniCPM-o teacher-forcing prefill range is invalid");
    }
    const auto exact = minicpmo45_prefill_segments(
        boundaries, 0, range_end, 0);
    const auto text = std::make_pair(text_begin, text_end);
    if (std::find(exact.begin(), exact.end(), text) == exact.end()) {
        throw std::runtime_error(
            "MiniCPM-o teacher-forcing splits omit the text span");
    }
    if (chunk_tokens == 0) return exact;

    std::vector<std::pair<int64_t, int64_t>> segments;
    const auto append = [&](const auto & parts) {
        segments.insert(segments.end(), parts.begin(), parts.end());
    };
    if (text_begin > 0) {
        append(minicpmo45_prefill_segments(
            boundaries, 0, text_begin, chunk_tokens));
    }
    segments.emplace_back(text);
    if (text_end < range_end) {
        append(minicpmo45_prefill_segments(
            boundaries, text_end, range_end, chunk_tokens));
    }
    return segments;
}

int run_minicpmo45_eval_batch(
        const std::string & model_path,
        const std::string & config_path,
        int64_t context_size,
        int64_t vision_batch_size) {
    mfq_tensor_backend::NoGradGuard no_grad;
    g_profiler.enabled = false;
    if (context_size <= 0) context_size = 8192;
    if (vision_batch_size <= 0) vision_batch_size = 16;
    auto runtime = MiniCPMO45Runtime::load(
        model_path, config_path, context_size);
    std::cout << "minicpmo45_eval_batch=ready"
              << " context_size=" << context_size
              << " vision_batch_size=" << vision_batch_size << std::endl;

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        auto request = nlohmann::json::parse(line, nullptr, false);
        if (!request.is_object()) {
            throw std::runtime_error(
                "MiniCPM-o eval request must be a JSON object");
        }
        if (request.value("shutdown", false)) break;
        const std::string request_id =
            request.at("request_id").get<std::string>();
        const std::string input_prefix =
            request.at("input_prefix").get<std::string>();
        const std::string output_prefix =
            request.value("output_prefix", std::string{});
        const int64_t question_count =
            request.at("question_count").get<int64_t>();
        const int64_t prefix_length =
            request.at("common_prefix_length").get<int64_t>();
        const bool reuse_prefix_cache =
            request.value("reuse_prefix_cache", true);
        const bool require_segmented_prefill =
            request.value("require_segmented_prefill", false);
        const int64_t segmented_prefill_chunk_tokens =
            request.value("segmented_prefill_chunk_tokens", int64_t{0});
        const int64_t max_new_tokens =
            request.value("max_new_tokens", int64_t{128});
        const double repetition_penalty =
            request.value("repetition_penalty", 1.02);
        const bool capture_logits =
            request.value("capture_logits", false);
        const bool ignore_eos =
            request.value("ignore_eos", false);
        const bool generate_tts =
            request.value("generate_tts", false);
        const bool tts_teacher_forcing =
            request.value("tts_teacher_forcing", false);
        const int64_t tts_max_steps =
            request.value("tts_max_steps", int64_t{2048});
        const double tts_temperature =
            request.value("tts_temperature", 0.8);
        const double tts_top_p =
            request.value("tts_top_p", 0.85);
        const int64_t tts_top_k =
            request.value("tts_top_k", int64_t{25});
        const double tts_min_p =
            request.value("tts_min_p", 0.01);
        const double tts_repetition_penalty =
            request.value("tts_repetition_penalty", 1.05);
        const int64_t tts_minimum_steps =
            request.value("tts_minimum_steps", int64_t{0});
        const int64_t tts_min_tokens_to_keep =
            request.value("tts_min_tokens_to_keep", int64_t{3});
        const int64_t seed =
            request.value("seed", int64_t{0});
        const auto eos_token_ids = request.at("eos_token_ids")
            .get<std::vector<int64_t>>();
        if (question_count <= 0 || prefix_length < 0 ||
                (reuse_prefix_cache && prefix_length <= 0) ||
                max_new_tokens <= 0 || repetition_penalty <= 0.0 ||
                segmented_prefill_chunk_tokens < 0 ||
                (segmented_prefill_chunk_tokens > 0 &&
                 !require_segmented_prefill) ||
                eos_token_ids.empty()) {
            throw std::runtime_error(
                "MiniCPM-o eval request has invalid generation settings");
        }
        if (generate_tts && (question_count != 1 || tts_max_steps <= 0 ||
                tts_temperature <= 0.0 ||
                tts_top_p <= 0.0 || tts_top_p > 1.0 ||
                tts_top_k < 3 || tts_top_k > 6562 ||
                tts_min_p < 0.0 || tts_min_p > 1.0 ||
                tts_repetition_penalty <= 0.0 ||
                tts_minimum_steps < 0 || tts_min_tokens_to_keep < 0 ||
                tts_min_tokens_to_keep > tts_top_k)) {
            throw std::runtime_error(
                "MiniCPM-o TTS batch requires one question and positive steps");
        }
        if (tts_teacher_forcing && !generate_tts) {
            throw std::runtime_error(
                "MiniCPM-o teacher forcing requires TTS generation");
        }
        const auto wall_started = std::chrono::steady_clock::now();
        auto pixels = minicpmo45_load_tensor(
            input_prefix + ".pixel_values.pt", false);
        auto patch_mask = minicpmo45_load_tensor(
            input_prefix + ".patch_mask.pt", false);
        auto target_sizes = minicpmo45_load_tensor(
            input_prefix + ".target_sizes.pt", false);
        if (target_sizes.defined()) {
            target_sizes = target_sizes.to(
                mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt64).contiguous();
        }
        auto audio_features = minicpmo45_load_tensor(
            input_prefix + ".audio_features.pt", false);
        auto audio_lengths = minicpmo45_load_tensor(
            input_prefix + ".audio_lengths.pt", false);
        if (pixels.defined() != patch_mask.defined() ||
                pixels.defined() != target_sizes.defined() ||
                (pixels.defined() &&
                 (pixels.dim() != 4 || pixels.size(1) != 3 ||
                  patch_mask.dim() != 2 || target_sizes.dim() != 2 ||
                  pixels.size(0) != patch_mask.size(0) ||
                  pixels.size(0) != target_sizes.size(0)))) {
            throw std::runtime_error(
                "MiniCPM-o eval tensors have incompatible shapes");
        }
        if (audio_features.defined() != audio_lengths.defined() ||
                (audio_features.defined() &&
                 (audio_features.dim() != 3 ||
                  audio_features.size(1) != 80 ||
                  audio_lengths.dim() != 1 ||
                  audio_lengths.size(0) != audio_features.size(0)))) {
            throw std::runtime_error(
                "MiniCPM-o eval batch audio tensors have incompatible shapes");
        }

        MFQ_CUDA_CHECK(cudaDeviceSynchronize());
        const auto vision_started = std::chrono::steady_clock::now();
        std::vector<mfq_tensor_backend::Tensor> vision_parts;
        std::vector<mfq_tensor_backend::Tensor> image_embedding_parts;
        mfq_tensor_backend::Tensor image_embeddings;
        if (pixels.defined() && pixels.size(0) > 0) {
            for (int64_t begin = 0; begin < pixels.size(0);
                    begin += vision_batch_size) {
                const int64_t count = std::min<int64_t>(
                    vision_batch_size, pixels.size(0) - begin);
                auto vision_part = runtime.vision.forward(
                    pixels.narrow(0, begin, count).to(mfq_tensor_backend::kCUDA),
                    patch_mask.narrow(0, begin, count),
                    target_sizes.narrow(0, begin, count));
                vision_parts.push_back(vision_part);
                image_embedding_parts.push_back(runtime.resampler.forward(
                    vision_part, target_sizes.narrow(0, begin, count)));
            }
            image_embeddings = mfq_tensor_backend::cat(
                image_embedding_parts, 0).contiguous();
        }
        mfq_tensor_backend::Tensor audio_embeddings;
        mfq_tensor_backend::Tensor audio_valid_lengths;
        double audio_seconds = 0.0;
        if (audio_features.defined()) {
            const auto audio_started = std::chrono::steady_clock::now();
            auto host_audio_lengths = audio_lengths.to(
                mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt64).contiguous();
            const auto * raw_lengths =
                host_audio_lengths.data_ptr<int64_t>();
            const auto valid_length_values =
                MiniCPMO45AudioEncoder::pooled_lengths(audio_lengths);
            audio_valid_lengths = mfq_tensor_backend::tensor(
                valid_length_values,
                mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64));
            const int64_t max_valid_length = *std::max_element(
                valid_length_values.begin(), valid_length_values.end());
            std::vector<mfq_tensor_backend::Tensor> audio_parts;
            audio_parts.reserve(static_cast<size_t>(audio_features.size(0)));
            for (int64_t index = 0; index < audio_features.size(0); ++index) {
                runtime.audio.reset();
                auto part_features = audio_features.index({
                    Slice(index, index + 1), Slice(),
                    Slice(0, raw_lengths[index])}).to(mfq_tensor_backend::kCUDA);
                auto part_length = mfq_tensor_backend::tensor(
                    std::vector<int64_t>{raw_lengths[index]},
                    mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64));
                auto part = runtime.audio.forward(
                    part_features, part_length, false);
                if (part.size(1) != valid_length_values[
                        static_cast<size_t>(index)]) {
                    throw std::runtime_error(
                        "MiniCPM-o per-segment Whisper length mismatch");
                }
                if (part.size(1) < max_valid_length) {
                    part = mfq_tensor_backend::constant_pad_nd(
                        part, {0, 0, 0, max_valid_length - part.size(1)}, 0.0);
                }
                audio_parts.push_back(part);
            }
            runtime.audio.reset();
            audio_embeddings = mfq_tensor_backend::cat(audio_parts, 0).contiguous();
            MFQ_CUDA_CHECK(cudaDeviceSynchronize());
            audio_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - audio_started).count();
        }
        MFQ_CUDA_CHECK(cudaDeviceSynchronize());
        const double vision_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - vision_started).count();
        pixels = mfq_tensor_backend::Tensor();
        patch_mask = mfq_tensor_backend::Tensor();
        vision_parts.clear();
        image_embedding_parts.clear();

        const auto question_prefix = [&](int64_t index) {
            std::ostringstream value;
            value << input_prefix << ".q" << std::setw(3)
                  << std::setfill('0') << index;
            return value.str();
        };
        const auto first_prefix = question_prefix(0);
        auto first_ids = minicpmo45_load_tensor(
            first_prefix + ".input_ids.pt", true)
            .to(mfq_tensor_backend::kCUDA, mfq_tensor_backend::kInt64).contiguous();
        if (first_ids.dim() == 1) first_ids = first_ids.unsqueeze(0);
        auto first_bounds = minicpmo45_load_tensor(
            first_prefix + ".image_bounds.pt", false);
        auto first_audio_bounds = minicpmo45_load_tensor(
            first_prefix + ".audio_bounds.pt", false);
        auto first_prefill_splits = minicpmo45_load_tensor(
            first_prefix + ".prefill_splits.pt",
            require_segmented_prefill);
        auto first_embeddings = minicpmo45_eval_embeddings(
            runtime, first_ids, image_embeddings, first_bounds,
            audio_embeddings, first_audio_bounds, audio_valid_lengths);
        if (prefix_length > first_ids.size(1)) {
            throw std::runtime_error(
                "MiniCPM-o common prefix exceeds the first prompt");
        }
        double prefix_seconds = 0.0;
        int64_t prefix_prefill_calls = 0;
        if (reuse_prefix_cache) {
            runtime.language.reset(1);
            auto prefix_ids = first_ids.narrow(1, 0, prefix_length);
            auto prefix_embeddings = first_embeddings.narrow(
                1, 0, prefix_length);
            auto prefix_mask = mfq_tensor_backend::ones(
                {1, prefix_length},
                mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA)
                    .dtype(mfq_tensor_backend::kBool));
            MFQ_CUDA_CHECK(cudaDeviceSynchronize());
            const auto prefix_started = std::chrono::steady_clock::now();
            mfq_tensor_backend::Tensor prefix_hidden;
            if (require_segmented_prefill) {
                const auto segments = minicpmo45_prefill_segments(
                    first_prefill_splits, 0, prefix_length,
                    segmented_prefill_chunk_tokens);
                for (const auto & segment : segments) {
                    const int64_t segment_begin = segment.first;
                    const int64_t segment_end = segment.second;
                    prefix_hidden = runtime.language.hidden_forward_inputs(
                        prefix_ids.narrow(
                            1, segment_begin, segment_end - segment_begin),
                        prefix_embeddings.narrow(
                            1, segment_begin, segment_end - segment_begin),
                        mfq_nullopt, mfq_nullopt, nullptr, mfq_nullopt);
                    ++prefix_prefill_calls;
                }
            } else {
                prefix_hidden = runtime.language.hidden_forward_inputs(
                    prefix_ids, prefix_embeddings,
                    mfq_nullopt, mfq_nullopt, nullptr, prefix_mask);
                prefix_prefill_calls = 1;
            }
            MFQ_CUDA_CHECK(cudaDeviceSynchronize());
            prefix_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - prefix_started).count();
            if (runtime.language.cache_pos != prefix_length) {
                throw std::runtime_error(
                    "MiniCPM-o prefix cache length mismatch");
            }
        }

        nlohmann::json results = nlohmann::json::array();
        for (int64_t question = 0; question < question_count; ++question) {
            const auto current_prefix = question_prefix(question);
            auto input_ids = question == 0
                ? first_ids
                : minicpmo45_load_tensor(
                    current_prefix + ".input_ids.pt", true)
                    .to(mfq_tensor_backend::kCUDA, mfq_tensor_backend::kInt64).contiguous();
            auto image_bounds = question == 0
                ? first_bounds
                : minicpmo45_load_tensor(
                    current_prefix + ".image_bounds.pt", false);
            auto audio_bounds = question == 0
                ? first_audio_bounds
                : minicpmo45_load_tensor(
                    current_prefix + ".audio_bounds.pt", false);
            auto attention_mask = minicpmo45_load_tensor(
                current_prefix + ".attention_mask.pt", true)
                .to(mfq_tensor_backend::kCUDA, mfq_tensor_backend::kBool).contiguous();
            auto tts_bound = minicpmo45_load_tensor(
                current_prefix + ".tts_bound.pt", tts_teacher_forcing);
            auto prefill_splits = minicpmo45_load_tensor(
                current_prefix + ".prefill_splits.pt",
                tts_teacher_forcing || require_segmented_prefill);
            if (input_ids.dim() == 1) input_ids = input_ids.unsqueeze(0);
            if (attention_mask.dim() == 1) {
                attention_mask = attention_mask.unsqueeze(0);
            }
            if (input_ids.size(0) != 1 ||
                    attention_mask.sizes() != input_ids.sizes() ||
                    (reuse_prefix_cache &&
                     prefix_length >= input_ids.size(1)) ||
                    input_ids.size(1) + max_new_tokens > context_size) {
                throw std::runtime_error(
                    "MiniCPM-o eval prompt exceeds the configured context");
            }
            auto input_embeddings = question == 0
                ? first_embeddings
                : minicpmo45_eval_embeddings(
                    runtime, input_ids, image_embeddings, image_bounds,
                    audio_embeddings, audio_bounds, audio_valid_lengths);
            if (reuse_prefix_cache) {
                runtime.language.cache_pos = prefix_length;
            } else {
                runtime.language.reset(1);
            }
            const int64_t evaluated_prefix_length =
                reuse_prefix_cache ? prefix_length : 0;
            const int64_t suffix_length =
                input_ids.size(1) - evaluated_prefix_length;
            auto suffix_ids = input_ids.narrow(
                1, evaluated_prefix_length, suffix_length);
            auto suffix_embeddings = input_embeddings.narrow(
                1, evaluated_prefix_length, suffix_length);
            MFQ_CUDA_CHECK(cudaDeviceSynchronize());
            const auto generation_started = std::chrono::steady_clock::now();
            mfq_tensor_backend::Tensor hidden;
            mfq_tensor_backend::Tensor teacher_text_hidden;
            mfq_tensor_backend::Tensor logits;
            int64_t suffix_prefill_calls = 0;
            if (tts_teacher_forcing) {
                if (reuse_prefix_cache || evaluated_prefix_length != 0) {
                    throw std::runtime_error(
                        "MiniCPM-o teacher forcing cannot reuse a prefix cache");
                }
                tts_bound = tts_bound.to(
                    mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt64).contiguous().reshape({-1});
                prefill_splits = prefill_splits.to(
                    mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt64).contiguous().reshape({-1});
                if (tts_bound.numel() != 2 || prefill_splits.numel() < 2) {
                    throw std::runtime_error(
                        "MiniCPM-o teacher-forcing boundaries are invalid");
                }
                const int64_t text_begin = tts_bound[0].item<int64_t>();
                const int64_t text_end = tts_bound[1].item<int64_t>();
                const auto segments = minicpmo45_teacher_prefill_segments(
                    prefill_splits, input_ids.size(1), text_begin, text_end,
                    segmented_prefill_chunk_tokens);
                bool found_text = false;
                for (const auto & segment : segments) {
                    const int64_t segment_begin = segment.first;
                    const int64_t segment_end = segment.second;
                    auto segment_hidden = runtime.language.hidden_forward_inputs(
                        input_ids.narrow(
                            1, segment_begin, segment_end - segment_begin),
                        input_embeddings.narrow(
                            1, segment_begin, segment_end - segment_begin),
                        mfq_nullopt, mfq_nullopt, nullptr, mfq_nullopt);
                    if (segment_begin == text_begin && segment_end == text_end) {
                        teacher_text_hidden = segment_hidden;
                        found_text = true;
                    }
                    hidden = segment_hidden;
                    ++suffix_prefill_calls;
                }
                if (!found_text) {
                    throw std::runtime_error(
                        "MiniCPM-o teacher-forcing splits omit the text span");
                }
            } else if (require_segmented_prefill) {
                const auto segments = minicpmo45_prefill_segments(
                    prefill_splits, evaluated_prefix_length,
                    input_ids.size(1), segmented_prefill_chunk_tokens);
                for (const auto & segment : segments) {
                    const int64_t segment_begin = segment.first;
                    const int64_t segment_end = segment.second;
                    hidden = runtime.language.hidden_forward_inputs(
                        input_ids.narrow(
                            1, segment_begin, segment_end - segment_begin),
                        input_embeddings.narrow(
                            1, segment_begin, segment_end - segment_begin),
                        mfq_nullopt, mfq_nullopt, nullptr, mfq_nullopt);
                    ++suffix_prefill_calls;
                }
                logits = runtime.language.logits_from_hidden(
                    hidden.index({Slice(), -1, Slice()}));
            } else {
                hidden = runtime.language.hidden_forward_inputs(
                    suffix_ids, suffix_embeddings,
                    mfq_nullopt, mfq_nullopt, nullptr, attention_mask);
                suffix_prefill_calls = 1;
                logits = runtime.language.logits_from_hidden(
                    hidden.index({Slice(), -1, Slice()}));
            }
            MFQ_CUDA_CHECK(cudaDeviceSynchronize());
            const double suffix_prefill_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - generation_started).count();
            const auto decode_started = std::chrono::steady_clock::now();
            double first_token_seconds = 0.0;
            auto counts = mfq_tensor_backend::zeros(
                {runtime.language.vocab_size()},
                mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA)
                    .dtype(mfq_tensor_backend::kInt32));
            std::vector<int64_t> generated;
            std::vector<mfq_tensor_backend::Tensor> captured_logits;
            std::vector<mfq_tensor_backend::Tensor> generated_hidden;
            generated.reserve(static_cast<size_t>(max_new_tokens));
            if (capture_logits) {
                captured_logits.reserve(
                    static_cast<size_t>(max_new_tokens));
            }
            for (int64_t step = 0;
                    !tts_teacher_forcing && step < max_new_tokens; ++step) {
                if (capture_logits) {
                    captured_logits.push_back(
                        logits.detach().to(mfq_tensor_backend::kCPU));
                }
                auto scores = generated.empty() || repetition_penalty == 1.0
                    ? logits
                    : sample_apply_penalties_cuda(
                        logits, counts, 0.0, 0.0,
                        repetition_penalty);
                auto next = sample_greedy_cuda(
                    scores.contiguous().view({1, -1}));
                const int64_t token = next.item<int64_t>();
                if (step == 0) {
                    first_token_seconds = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - decode_started).count();
                }
                generated.push_back(token);
                if ((!ignore_eos &&
                        minicpmo45_contains_token(eos_token_ids, token)) ||
                        step + 1 == max_new_tokens) {
                    break;
                }
                sample_token_counts_add_cuda(counts, next.contiguous());
                auto next_hidden = runtime.language.hidden_forward(
                    next.reshape({1, 1}));
                generated_hidden.push_back(next_hidden);
                logits = runtime.language.logits_from_hidden(
                    next_hidden.index({Slice(), -1, Slice()}));
            }
            mfq_tensor_backend::Tensor tts_codes;
            if (generate_tts) {
                mfq_tensor_backend::Tensor text_ids;
                mfq_tensor_backend::Tensor text_hidden;
                if (tts_teacher_forcing) {
                    const int64_t begin =
                        tts_bound[0].item<int64_t>();
                    const int64_t end =
                        tts_bound[1].item<int64_t>();
                    if (begin < evaluated_prefix_length || end <= begin ||
                            end > input_ids.size(1)) {
                        throw std::runtime_error(
                            "MiniCPM-o teacher-forcing span is invalid");
                    }
                    text_ids = input_ids.narrow(1, begin, end - begin);
                    text_hidden = teacher_text_hidden;
                } else {
                    if (generated_hidden.empty()) {
                        throw std::runtime_error(
                            "MiniCPM-o TTS generation produced no text condition");
                    }
                    constexpr int64_t kTtsTextEosToken = 151704;
                    const auto text_eos = std::find(
                        generated.begin(), generated.end(), kTtsTextEosToken);
                    const int64_t text_length = std::min<int64_t>(
                        static_cast<int64_t>(generated_hidden.size()),
                        static_cast<int64_t>(
                            std::distance(generated.begin(), text_eos)));
                    if (text_length <= 0) {
                        throw std::runtime_error(
                            "MiniCPM-o TTS generation produced no text tokens");
                    }
                    text_ids = mfq_tensor_backend::tensor(
                        std::vector<int64_t>(
                            generated.begin(), generated.begin() + text_length),
                        mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA)
                            .dtype(mfq_tensor_backend::kInt64)).reshape({1, text_length});
                    text_hidden = mfq_tensor_backend::cat(generated_hidden, 1)
                        .narrow(1, 0, text_length).contiguous();
                }
                auto condition = runtime.tts.condition(
                    text_ids, text_hidden);
                std::mt19937 evaluator_rng(
                    static_cast<std::mt19937::result_type>(seed));
                tts_codes = runtime.tts.generate_official(
                    condition, tts_max_steps, 6561,
                    tts_minimum_steps,
                    tts_temperature, tts_top_p, tts_top_k,
                    tts_repetition_penalty, nullptr, tts_min_p,
                    &evaluator_rng, tts_min_tokens_to_keep)
                    .to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt64).reshape({-1});
                if (!output_prefix.empty()) {
                    std::ostringstream path;
                    path << output_prefix << ".q" << std::setw(3)
                         << std::setfill('0') << question << ".tts_codes.pt";
                    minicpmo45_save_tensor(tts_codes, path.str());
                }
            }
            MFQ_CUDA_CHECK(cudaDeviceSynchronize());
            const double generation_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - generation_started).count();
            const double decode_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - decode_started).count();
            if (capture_logits && !output_prefix.empty()) {
                std::ostringstream path;
                path << output_prefix << ".q" << std::setw(3)
                     << std::setfill('0') << question << ".logits.pt";
                minicpmo45_save_tensor(
                    mfq_tensor_backend::cat(captured_logits, 0), path.str());
            }
            results.push_back({
                {"question_index", question},
                {"context_tokens", input_ids.size(1)},
                {"prefix_tokens", evaluated_prefix_length},
                {"suffix_tokens", suffix_length},
                {"suffix_prefill_calls", suffix_prefill_calls},
                {"generated_token_ids", generated},
                {"generation_seconds", generation_seconds},
                {"suffix_prefill_seconds", suffix_prefill_seconds},
                {"first_token_seconds", first_token_seconds},
                {"decode_seconds", decode_seconds},
                {"tts_code_count", tts_codes.defined()
                    ? tts_codes.numel() : 0},
            });
            if (question == 0) {
                first_ids = mfq_tensor_backend::Tensor();
                first_bounds = mfq_tensor_backend::Tensor();
                first_audio_bounds = mfq_tensor_backend::Tensor();
                first_embeddings = mfq_tensor_backend::Tensor();
            }
        }
        MFQ_CUDA_CHECK(cudaDeviceSynchronize());
        const auto stats = mfq_cuda_memory_stats(mfq_current_cuda_device());
        nlohmann::json response = {
            {"request_id", request_id},
            {"status", "ok"},
            {"frame_count", image_embeddings.defined()
                ? image_embeddings.size(0) : 0},
            {"audio_count", audio_embeddings.defined()
                ? audio_embeddings.size(0) : 0},
            {"vision_seconds", vision_seconds},
            {"audio_seconds", audio_seconds},
            {"prefix_seconds", prefix_seconds},
            {"prefix_prefill_calls", prefix_prefill_calls},
            {"segmented_prefill_chunk_tokens",
                segmented_prefill_chunk_tokens},
            {"wall_seconds", std::chrono::duration<double>(
                std::chrono::steady_clock::now() - wall_started).count()},
            {"allocated_bytes", stats.allocated_bytes},
            {"reserved_bytes", stats.reserved_bytes},
            {"peak_allocated_bytes", stats.peak_allocated_bytes},
            {"peak_reserved_bytes", stats.peak_reserved_bytes},
            {"questions", results},
        };
        std::cout << "minicpmo45_eval_result="
                  << response.dump() << std::endl;
    }
    return 0;
}

int run_minicpmo45_composite(
        const std::string & model_path,
        const std::string & config_path,
        const std::string & input_prefix,
        const std::string & output_prefix,
        int64_t context_size,
        int64_t tts_steps) {
    mfq_tensor_backend::NoGradGuard no_grad;
    g_profiler.enabled = false;
    if (input_prefix.empty() || output_prefix.empty()) {
        throw std::runtime_error(
            "MiniCPM-o composite graph requires input and output prefixes");
    }
    auto runtime = MiniCPMO45Runtime::load(
        model_path, config_path, context_size);
    const auto input_ids = minicpmo45_load_tensor(
        input_prefix + ".input_ids.pt", true);
    const auto position_ids = minicpmo45_load_tensor(
        input_prefix + ".position_ids.pt", false);
    const auto attention_mask = minicpmo45_load_tensor(
        input_prefix + ".attention_mask.pt", false);
    const auto pixels = minicpmo45_load_tensor(
        input_prefix + ".pixel_values.pt", false);
    const auto patch_mask = minicpmo45_load_tensor(
        input_prefix + ".patch_mask.pt", false);
    const auto target_sizes = minicpmo45_load_tensor(
        input_prefix + ".target_sizes.pt", false);
    const auto image_bounds = minicpmo45_load_tensor(
        input_prefix + ".image_bounds.pt", false);
    const auto audio_features = minicpmo45_load_tensor(
        input_prefix + ".audio_features.pt", false);
    const auto audio_lengths = minicpmo45_load_tensor(
        input_prefix + ".audio_lengths.pt", false);
    const auto audio_bounds = minicpmo45_load_tensor(
        input_prefix + ".audio_bounds.pt", false);
    auto result = runtime.forward(
        input_ids, position_ids, attention_mask,
        pixels, patch_mask, target_sizes, image_bounds,
        audio_features, audio_lengths, audio_bounds);
    minicpmo45_save_tensor(
        result.vision_states, output_prefix + ".vision_states.pt");
    minicpmo45_save_tensor(
        result.image_embeddings, output_prefix + ".image_embeddings.pt");
    minicpmo45_save_tensor(
        result.audio_embeddings, output_prefix + ".audio_embeddings.pt");
    minicpmo45_save_tensor(
        result.input_embeddings, output_prefix + ".input_embeddings.pt");
    minicpmo45_save_tensor(
        result.hidden_states, output_prefix + ".hidden_states.pt");
    minicpmo45_save_tensor(
        result.logits, output_prefix + ".logits.pt");
    if (tts_steps > 0) {
        mfq_tensor_backend::Tensor condition = minicpmo45_load_tensor(
            input_prefix + ".tts_inputs_embeds.pt", false);
        if (!condition.defined()) {
            auto bound = minicpmo45_load_tensor(
                input_prefix + ".tts_bound.pt", true)
                .to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt64).contiguous();
            if (bound.dim() != 1 || bound.numel() != 2) {
                throw std::runtime_error(
                    "MiniCPM-o tts_bound must contain [begin,end]");
            }
            const int64_t begin = bound[0].item<int64_t>();
            const int64_t end = bound[1].item<int64_t>();
            if (begin < 0 || end <= begin ||
                    end > input_ids.size(-1) || input_ids.size(0) != 1) {
                throw std::runtime_error(
                    "MiniCPM-o TTS text span is invalid");
            }
            condition = runtime.tts.condition(
                input_ids.index({0, Slice(begin, end)}),
                result.hidden_states.index({0, Slice(begin, end), Slice()}));
        } else {
            condition = condition.to(mfq_tensor_backend::kCUDA).contiguous();
        }
        std::vector<mfq_tensor_backend::Tensor> tts_logits;
        const auto codes = runtime.tts.generate_official(
            condition, tts_steps, 6561,
            50,
            0.8, 0.85, 25, 1.05, &tts_logits);
        minicpmo45_save_tensor(
            condition, output_prefix + ".tts_inputs_embeds.pt");
        minicpmo45_save_tensor(
            codes, output_prefix + ".tts_codes.pt");
        if (!tts_logits.empty()) {
            minicpmo45_save_tensor(
                mfq_tensor_backend::stack(tts_logits, 1),
                output_prefix + ".tts_logits.pt");
        }
    }
    std::cout << "minicpmo45_composite=ok"
              << " batch=" << input_ids.size(0)
              << " tokens=" << input_ids.size(1)
              << " images=" << (image_bounds.defined() ? image_bounds.size(0) : 0)
              << " audios=" << (audio_bounds.defined() ? audio_bounds.size(0) : 0)
              << " tts_steps=" << tts_steps << "\n";
    return 0;
}
