"""Executable ownership rules for shared native-runtime behavior."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METAL = ROOT / "cpp_runtime" / "backends" / "metal"
MODELS = METAL / "models"
MTP_HEADER = (METAL / "runtime" / "mlx_mtp.h").read_text(encoding="utf-8")
MTP_SOURCE = (METAL / "runtime" / "mlx_mtp.cpp").read_text(encoding="utf-8")
SAMPLING_HEADER = (METAL / "runtime" / "mlx_sampling.h").read_text(
    encoding="utf-8"
)
TRANSFORMER_HEADER = (METAL / "runtime" / "mlx_transformer.h").read_text(
    encoding="utf-8"
)
QWEN = (MODELS / "qwen35" / "mlx_qwen35_causal_lm.cpp").read_text(
    encoding="utf-8"
)
QWEN4 = (MODELS / "flash_next" / "mlx_qwen4_causal_lm.cpp").read_text(
    encoding="utf-8"
)
QWEN4_HEADER = (
    MODELS / "flash_next" / "mlx_qwen4_causal_lm.h"
).read_text(encoding="utf-8")
DSV = (MODELS / "deepseek_v4" / "mlx_deepseek_v4_causal_lm.cpp").read_text(
    encoding="utf-8"
)
DSV_MOE = (MODELS / "deepseek_v4" / "mlx_deepseek_v4_moe.cpp").read_text(
    encoding="utf-8"
)
DSV41 = (
    MODELS / "deepseek_v41" / "mlx_deepseek_v41_causal_lm.cpp"
).read_text(encoding="utf-8")
DSV_DSPARK = (
    MODELS / "deepseek_v4" / "mlx_deepseek_v4_dspark.cpp"
).read_text(encoding="utf-8")
DSV41_DSPARK = (
    MODELS / "deepseek_v41" / "mlx_deepseek_v41_dspark.cpp"
).read_text(encoding="utf-8")
DSV41_MOE = (
    MODELS / "deepseek_v41" / "mlx_deepseek_v41_moe.cpp"
).read_text(encoding="utf-8")
DSV41_ENGRAM = (
    MODELS / "deepseek_v41" / "mlx_deepseek_v41_engram.cpp"
).read_text(encoding="utf-8")
MXFP8_ROW_STORE = (
    METAL / "storage" / "mlx_mxfp8_row_store.cpp"
).read_text(encoding="utf-8")
SERVER_COMPONENTS = (
    METAL / "runtime" / "mlx_server_components.cpp"
).read_text(encoding="utf-8")
CONTRIBUTING = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
CUDA_MTP_HEADER = (
    ROOT / "cpp_runtime" / "engine" / "include" / "mtp_policy.h"
).read_text(encoding="utf-8")
CUDA_APP = (
    ROOT / "cpp_runtime" / "backends" / "cuda" / "apps" / "runtime_main.cpp"
).read_text(encoding="utf-8")
CUDA_RUNTIME_COMMAND = (
    ROOT / "cpp_runtime" / "backends" / "cuda" / "commands" / "runtime.cpp"
).read_text(encoding="utf-8")
CUDA_CLI = (
    ROOT / "cpp_runtime" / "backends" / "cuda" / "commands" / "cli.h"
).read_text(encoding="utf-8")
CUDA_MODELS = ROOT / "cpp_runtime" / "backends" / "cuda" / "models"
CUDA_OPS = ROOT / "cpp_runtime" / "backends" / "cuda" / "ops"
CUDA_RUNTIME = ROOT / "cpp_runtime" / "backends" / "cuda" / "engine"
CUDA_ENGINE_SOURCE = (CUDA_RUNTIME / "cuda_engine.cpp").read_text(
    encoding="utf-8"
)
CUDA_RUNTIME_SOURCE = "\n".join(
    (CUDA_RUNTIME / name).read_text(encoding="utf-8")
    for name in (
        "cuda_engine.cpp",
        "generation.cpp",
        "text_session_cache.cpp",
        "options.cpp",
    )
)
CUDA_MTP_SOURCE = (CUDA_RUNTIME / "mtp.cpp").read_text(encoding="utf-8")
CUDA_DECODE = CUDA_APP + "\n" + CUDA_RUNTIME_SOURCE
CUDA_BACKEND_SOURCE = "\n".join(
    path.read_text(encoding="utf-8")
    for path in (ROOT / "cpp_runtime" / "backends" / "cuda").rglob("*")
    if path.suffix in {".h", ".cpp"}
)
CUDA_REGISTRY = (CUDA_MODELS / "registry.cpp").read_text(encoding="utf-8")
CUDA_TRANSFORMER_LOADER = (
    CUDA_MODELS / "transformer.cpp"
).read_text(encoding="utf-8")
CUDA_TRANSFORMER_HEADER = (
    CUDA_MODELS / "transformer.h"
).read_text(encoding="utf-8")
CUDA_TRANSFORMER_PARTS = {
    name: (CUDA_MODELS / f"{name}.cpp").read_text(encoding="utf-8")
    for name in ("rope", "ffn", "kv_cache", "full_block")
}
CUDA_QWEN_LINEAR = (
    CUDA_MODELS / "qwen35" / "linear_attention.h"
).read_text(encoding="utf-8") + (\
    CUDA_MODELS / "qwen35" / "causal_lm.cpp"
).read_text(encoding="utf-8")
CUDA_CAUSAL_LM = (CUDA_MODELS / "causal_lm.h").read_text(
    encoding="utf-8"
)
CUDA_CAUSAL_LM_SOURCE = (CUDA_MODELS / "causal_lm.cpp").read_text(
    encoding="utf-8"
)
CUDA_CAUSAL_LM_IMPL = (CUDA_MODELS / "causal_lm_impl.h").read_text(
    encoding="utf-8"
)
CUDA_MODEL_HEADERS = "\n".join(
    path.read_text(encoding="utf-8")
    for path in CUDA_MODELS.rglob("*.h")
)
CUDA_MODEL_FINALIZERS = {
    name: (CUDA_MODELS / name / "causal_lm.cpp").read_text(encoding="utf-8")
    for name in ("qwen4_exp", "glm5_next", "deepseek_v4", "deepseek_v41")
}
CUDA_QWEN_BATCH_HEADER = (
    CUDA_RUNTIME / "cuda_batching.h"
).read_text(encoding="utf-8")
CUDA_QWEN_BATCH_SOURCE = (
    CUDA_RUNTIME / "cuda_batching.cpp"
).read_text(encoding="utf-8")
CUDA_QWEN_BATCH_STATE = "\n".join(
    (CUDA_MODELS / "qwen35" / name).read_text(encoding="utf-8")
    for name in ("batch_state.h", "batch_state.cpp")
)
CUDA_CAUSAL_LM_LOADER = (CUDA_MODELS / "loader.cpp").read_text(
    encoding="utf-8"
)
SPARSE_OPERATOR = (METAL / "ops" / "mlx_sparse_attention.cpp").read_text(
    encoding="utf-8"
)
SPARSE_HEADER = (METAL / "ops" / "mlx_sparse_attention.h").read_text(
    encoding="utf-8"
)
DSA_OPERATOR = (METAL / "ops" / "mlx_dsa.cpp").read_text(encoding="utf-8")
DSA_HEADER = (METAL / "ops" / "mlx_dsa.h").read_text(encoding="utf-8")
MX_OPERATOR = (METAL / "ops" / "mlx_mx.cpp").read_text(encoding="utf-8")
MX_HEADER = (METAL / "ops" / "mlx_mx.h").read_text(encoding="utf-8")
MOE_OPERATOR = (METAL / "ops" / "mlx_moe.cpp").read_text(encoding="utf-8")
DECODE_APP = (METAL / "apps" / "mfq_decode_mlx.cpp").read_text(
    encoding="utf-8"
)
PLATFORM = (METAL / "runtime" / "mlx_platform.h").read_text(encoding="utf-8")
CORE = ROOT / "cpp_runtime" / "core"
MODEL_SOURCE_HEADER = (
    CORE / "include" / "mfq" / "model_source.h"
).read_text(encoding="utf-8")
MFQ_SOURCE = (CORE / "mfq_model_source.cpp").read_text(encoding="utf-8")
HF_SOURCE = (CORE / "hf_safetensors_source.cpp").read_text(encoding="utf-8")
HF_MODEL_SOURCE = (CORE / "hf_model_source.cpp").read_text(encoding="utf-8")
MODEL_SOURCE_FACTORY = (CORE / "model_source.cpp").read_text(encoding="utf-8")
METAL_CONTAINER = (
    METAL / "storage" / "mfq_container.cpp"
).read_text(encoding="utf-8")
CUDA_MFE_STORE = (
    ROOT / "cpp_runtime" / "backends" / "cuda" / "storage" / "mfe_expert_store.cpp"
).read_text(encoding="utf-8")


def model_sources() -> str:
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in MODELS.rglob("*")
        if path.suffix in {".h", ".cpp"}
    )


def test_cuda_ops_do_not_depend_on_engine_or_parse_environment() -> None:
    for path in CUDA_OPS.rglob("*"):
        if path.suffix not in {".h", ".cpp", ".cu"}:
            continue
        source = path.read_text(encoding="utf-8")
        assert not re.search(r'#include\s*[<"](?:\.\./)*engine/', source)
        if path.name != "cuda_execution.cpp":
            assert "getenv(" not in source


def test_cuda_models_and_storage_do_not_parse_execution_environment() -> None:
    for directory in (CUDA_MODELS, CUDA_RUNTIME.parent / "storage"):
        for path in directory.rglob("*"):
            if path.suffix in {".h", ".cpp", ".cu"}:
                assert "getenv(" not in path.read_text(encoding="utf-8")


def test_cuda_models_do_not_depend_on_cuda_engine_implementation() -> None:
    for path in CUDA_MODELS.rglob("*"):
        if path.suffix not in {".h", ".cpp", ".cu"}:
            continue
        source = path.read_text(encoding="utf-8")
        assert not re.search(r'#include\s+"(?:\.\./)*engine/', source)
        for header in (
            "components.h",
            "cuda_batching.h",
            "cuda_engine.h",
            "decode_graph.h",
            "generation.h",
            "model_loader.h",
            "options.h",
            "text_session_cache.h",
        ):
            assert f'#include "{header}"' not in source
        assert "mfq::engine::Engine" not in source
        assert "engine_binder" not in source


def test_cuda_format_operators_only_receive_the_profiler() -> None:
    ops = CUDA_RUNTIME.parent / "ops"
    for name in ("nint.cpp", "vq.cpp", "include/nint.h", "include/vq.h"):
        source = (ops / name).read_text(encoding="utf-8")
        assert "CudaProfiler&" in source
        assert "CudaExecutionContext" not in source
        assert "cuda_execution_context(" not in source
        assert "KlMmqMode" not in source


def test_cuda_leaf_ops_receive_only_their_required_resources() -> None:
    quant_common = (CUDA_OPS / "include" / "quant_linear_common.h").read_text(
        encoding="utf-8"
    )
    mixed_moe = (CUDA_OPS / "include" / "mixed_moe.h").read_text(
        encoding="utf-8"
    )
    nint_group = (CUDA_OPS / "include" / "nint_linear_group.h").read_text(
        encoding="utf-8"
    )
    moe_types = (CUDA_OPS / "include" / "moe_types.h").read_text(
        encoding="utf-8"
    )

    assert "CudaExecutionContext" not in quant_common
    assert "CudaExecutionContext" not in mixed_moe
    assert "CudaExecutionContext" not in nint_group
    assert "CudaProfiler& profiler, KlMmqState& kl_mmq" in quant_common
    assert "ModelParallelCollectiveRuntime& collectives" in quant_common
    for helper in ("moe_tensor_to_device", "moe_route_to_device"):
        signature = moe_types.split(helper, 1)[1].split(");", 1)[0]
        assert "ModelParallelCollectiveRuntime& collectives" in signature
        assert "CudaExecutionContext" not in signature


def test_cuda_quant_runtime_implementations_stay_out_of_headers() -> None:
    headers = {
        "mixed_moe.h": 220,
        "quant_linear_groups.h": 170,
        "quant_linear_weight.h": 120,
    }
    for name, limit in headers.items():
        source = (CUDA_OPS / "include" / name).read_text(encoding="utf-8")
        assert len(source.splitlines()) < limit

    moe = (CUDA_OPS / "moe.cpp").read_text(encoding="utf-8")
    quant = (CUDA_OPS / "quant_linear.cpp").read_text(encoding="utf-8")
    assert "MixedMoeRuntime::forward(" in moe
    assert "QuantLinear::forward(" in quant
    assert "QuantLinearGroup::forward(" in quant


def test_cuda_transformer_header_stays_declarative() -> None:
    assert len(CUDA_TRANSFORMER_HEADER.splitlines()) < 30
    assert '#include "full_block.h"' not in CUDA_TRANSFORMER_HEADER
    limits = {"rope": 60, "ffn": 120, "kv_cache": 80, "full_block": 100}
    for name, limit in limits.items():
        header = (CUDA_MODELS / f"{name}.h").read_text(encoding="utf-8")
        assert len(header.splitlines()) < limit

    implementations = {
        "rope": ("RopeCache::RopeCache(",),
        "ffn": ("FFN::forward(", "FFN::forward_impl("),
        "kv_cache": ("KVCache::KVCache(", "KVCache::paged_view("),
        "full_block": ("Block::forward_context(", "FullBlock::forward_impl("),
    }
    for name, symbols in implementations.items():
        header = (CUDA_MODELS / f"{name}.h").read_text(encoding="utf-8")
        for symbol in symbols:
            assert symbol in CUDA_TRANSFORMER_PARTS[name]
            assert symbol not in header


def test_cuda_dsv4_projection_and_moe_loading_are_shared() -> None:
    ffn = (CUDA_MODELS / "ffn.cpp").read_text(encoding="utf-8")
    v4 = (CUDA_MODELS / "deepseek_v4" / "causal_lm.cpp").read_text(
        encoding="utf-8"
    )
    headers = "\n".join(
        (CUDA_MODELS / name / "causal_lm.h").read_text(encoding="utf-8")
        for name in ("deepseek_v4", "deepseek_v41")
    )
    loaders = [
        (CUDA_MODELS / path).read_text(encoding="utf-8")
        for path in (
            "qwen35/causal_lm.cpp",
            "deepseek_v4/causal_lm.cpp",
            "deepseek_v41/causal_lm.cpp",
            "glm_dsa/causal_lm.cpp",
        )
    ]

    assert v4.count("nint_matmul_groupwise_u8(") == 1
    assert "nint_matmul_groupwise_u8(" not in headers
    assert "FFN load_moe_weights(" in ffn
    assert all("load_moe_weights(" in source for source in loaders)


def test_cuda_native_tensor_ops_stay_split_by_domain() -> None:
    sources = {
        name: (CUDA_RUNTIME.parent / "src" / f"mfq_native_tensor_{name}.cu")
        for name in (
            "ops", "blas", "creation", "indexing", "reduction", "signal", "sort"
        )
    }
    assert all(len(path.read_text(encoding="utf-8").splitlines()) < 1000
               for path in sources.values())
    assert "Tensor matmul(" in sources["blas"].read_text(encoding="utf-8")
    assert "Tensor index_select_cuda(" in sources["indexing"].read_text(encoding="utf-8")
    assert "Tensor reduce_cuda(" in sources["reduction"].read_text(encoding="utf-8")
    assert "Tensor conv1d(" in sources["signal"].read_text(encoding="utf-8")
    assert "std::tuple<Tensor, Tensor> topk(" in sources["sort"].read_text(encoding="utf-8")


def test_model_sources_are_backend_neutral_and_shared() -> None:
    assert "class ModelSource" in MODEL_SOURCE_HEADER
    assert "class MfqModelSource final : public ModelSource" in (
        CORE / "include" / "mfq" / "mfq_model_source.h"
    ).read_text(encoding="utf-8")
    assert "class HfSafetensorsSource final : public ModelSource" in (
        CORE / "include" / "mfq" / "hf_safetensors_source.h"
    ).read_text(encoding="utf-8")
    assert "class HfModelSource final : public ModelSource" in (
        CORE / "include" / "mfq" / "hf_model_source.h"
    ).read_text(encoding="utf-8")
    for source in (
        MODEL_SOURCE_HEADER,
        MFQ_SOURCE,
        HF_SOURCE,
        HF_MODEL_SOURCE,
    ):
        assert "backends/" not in source
        assert "mlx::" not in source
        assert "#include <cuda" not in source.lower()
        assert "#include <metal" not in source.lower()
    assert '#include "mfq/mfq_model_source.h"' not in CUDA_BACKEND_SOURCE
    assert "HfModelSource" not in CUDA_BACKEND_SOURCE
    assert '#include "mfq/mfq_model_source.h"' in MODEL_SOURCE_FACTORY
    assert "HfModelSource" in MODEL_SOURCE_FACTORY
    assert '#include "mfq/mfq_model_source.h"' in METAL_CONTAINER
    assert "HfModelSource" in METAL_CONTAINER
    assert "bad MFQ magic" not in CUDA_BACKEND_SOURCE
    assert "MfqContainer::load_records" not in METAL_CONTAINER
    assert "HfSafetensorStore" not in METAL_CONTAINER
    assert "MXT1" not in METAL_CONTAINER
    assert "struct MfqFile" not in CUDA_BACKEND_SOURCE
    assert "open_model_source" in CUDA_BACKEND_SOURCE
    assert "const mfq::ModelSource" in CUDA_BACKEND_SOURCE
    assert "direct_source" not in CUDA_BACKEND_SOURCE
    assert 'record.metadata.dtype = "FP8-128SQ"' in HF_MODEL_SOURCE
    assert "record_.read_range(offset, destination)" in CUDA_MFE_STORE


def test_native_cli_uses_backend_neutral_model_and_tokenizer_options() -> None:
    sources = (
        CUDA_APP + "\n" + CUDA_RUNTIME_COMMAND + "\n" + CUDA_CLI,
        DECODE_APP,
        (METAL / "apps" / "mfq_perplexity_mlx.cpp").read_text(encoding="utf-8"),
    )
    for source in sources:
        assert '"--model"' in source
        assert '"--tokenizer"' in source
        assert '"--mfq"' not in source
        assert '"--tokenizer-model"' not in source
        assert '"--tokenizer-gguf"' not in source


def test_cuda_cli_is_a_thin_client_of_the_runtime_library() -> None:
    apps = ROOT / "cpp_runtime" / "backends" / "cuda" / "apps"
    for name in ("runtime", "diagnostics", "eval"):
        source = (apps / f"{name}_main.cpp").read_text(encoding="utf-8")
        assert f"mfq::cuda::commands::run_{name}(argc, argv)" in source
        assert "argv[" not in source
        assert "struct Model" not in source

    cmake = (ROOT / "cpp_runtime" / "backends" / "cuda" / "CMakeLists.txt").read_text(
        encoding="utf-8"
    )
    assert "add_library(mfq-cuda-runtime STATIC" in cmake
    assert not (CUDA_RUNTIME / "cuda_runtime.cpp").exists()
    assert "engine/cuda_runtime.cpp" not in cmake
    assert "engine/cuda_engine.cpp" in cmake
    assert "engine/generation.cpp" in cmake
    assert "engine/text_session_cache.cpp" in cmake
    assert "add_executable(mfq-runtime\n" in cmake
    torch_sources = cmake.split("add_executable(mfq-runtime-torch", 1)[1].split(")", 1)[0]
    assert "${MFQ_CUDA_ROOT}/commands/minicpmo45.cpp" in torch_sources
    assert "${MFQ_CUDA_ROOT}/models/minicpmo45/components.cpp" in torch_sources
    assert "mfq-cuda-runtime mfq-runtime-communication" in cmake
    assert "${MFQ_CUDA_ROOT}/commands/runtime.cpp" in cmake
    assert "${MFQ_CUDA_ROOT}/commands/diagnostics.cpp" in cmake
    assert "${MFQ_CUDA_ROOT}/commands/eval.cpp" in cmake
    assert ("mfq-" + "decode") not in cmake
    assert "execute_runtime" in CUDA_RUNTIME_COMMAND
    assert "RuntimeExecution" not in CUDA_RUNTIME_SOURCE
    assert "run_runtime(RuntimeOptions" not in CUDA_RUNTIME_SOURCE
    assert '"--server"' not in CUDA_RUNTIME_SOURCE
    assert '"--stdio"' not in CUDA_RUNTIME_SOURCE
    assert "MFQ_SERVER_" not in CUDA_RUNTIME_SOURCE


def test_cuda_runtime_has_one_shared_generation_path() -> None:
    generation = (CUDA_RUNTIME / "generation.cpp").read_text(encoding="utf-8")
    header = (CUDA_RUNTIME / "generation.h").read_text(encoding="utf-8")
    shared = (
        ROOT / "cpp_runtime" / "engine" / "include" / "inference.h"
    ).read_text(encoding="utf-8")

    assert "generate_tokens" not in generation + header
    assert "MFQ_RUNTIME_QWEN38_TEXT_FLOW" not in CUDA_ENGINE_SOURCE
    assert "mfq::cuda::internal::generate(" in CUDA_ENGINE_SOURCE
    assert "mfq::engine::generate(" in generation
    assert "mfq::engine::generate_target(" in generation
    assert "while (generated < max_tokens)" not in generation
    assert "generate_cli_tokens" not in header
    assert not (
        CUDA_RUNTIME.parent / "commands" / "token_generation.h"
    ).exists()
    assert not (
        CUDA_RUNTIME.parent / "commands" / "token_generation.cpp"
    ).exists()
    assert "generate_diagnostic_tokens" in (
        CUDA_RUNTIME.parent / "diagnostics" / "token_generation.h"
    ).read_text(encoding="utf-8")
    assert '"--ids"' not in CUDA_RUNTIME_COMMAND
    assert '"--ids-file"' not in CUDA_RUNTIME_COMMAND
    assert '"--gen"' not in CUDA_RUNTIME_COMMAND
    assert "run_token_mode" not in CUDA_RUNTIME_COMMAND
    assert "generate_target(" in shared
    assert "InferenceRequest" in shared
    assert "TextGeneration" not in shared
    assert not (
        ROOT / "cpp_runtime" / "engine" / "include" / "text_generation.h"
    ).exists()
    assert not (
        ROOT / "cpp_runtime" / "engine" / "src" / "text_generation.cpp"
    ).exists()
    assert "ensure_captured(" in generation


def test_cuda_runtime_hides_model_session_and_batch_implementation() -> None:
    execution = (CUDA_OPS / "include" / "cuda_execution.h").read_text(encoding="utf-8")
    options = (CUDA_RUNTIME / "options.cpp").read_text(encoding="utf-8")

    assert "struct CudaExecutionContext" in execution
    assert "class CudaProfilerAccess" not in execution
    assert "inline const CudaProfilerAccess g_profiler" not in execution
    assert "extern CudaProfiler&" not in execution
    assert "CudaExecutionContext& execution;" in CUDA_ENGINE_SOURCE
    assert "std::make_shared<CudaExecutionContext>()" in CUDA_ENGINE_SOURCE
    execution_source = (CUDA_OPS / "cuda_execution.cpp").read_text(
        encoding="utf-8"
    )
    assert "static CudaExecutionContext context" not in execution_source
    assert "thread_local bool g_decode_graph" not in CUDA_BACKEND_SOURCE
    command_and_diagnostic_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for directory in (
            CUDA_RUNTIME.parent / "commands",
            CUDA_RUNTIME.parent / "diagnostics",
        )
        for path in directory.glob("*.cpp")
    )
    assert "cuda_execution_context()" not in command_and_diagnostic_sources
    for alias in (
        "g_tensor_parallel",
        "g_expert_parallel",
        "g_model_parallel_collectives",
        "g_layer_placement",
        "g_moe_expert_cache",
        "g_dsv4_cpu_offload_layers",
        "g_dense_cpu_layer_count",
        "g_loading_cpu_layer",
    ):
        assert alias not in execution
    assert "class PagedSessionBindings" in (
        ROOT / "cpp_runtime" / "engine" / "include" /
        "paged_session_bindings.h"
    ).read_text(encoding="utf-8")
    assert "struct CompactDistribution" in CUDA_MTP_HEADER
    assert "struct CompactDistribution" not in CUDA_MTP_SOURCE
    constraint = (
        ROOT / "cpp_runtime" / "engine" / "src" / "token_constraint.cpp"
    ).read_text(encoding="utf-8")
    transport = (
        ROOT / "cpp_runtime" / "transport" / "src" / "common.cpp"
    ).read_text(encoding="utf-8")
    assert "class GrammarConstraint" in constraint
    assert "class MfqGrammarConstraint" not in transport
    assert "execution.reset();" in options
    assert "CudaExecutionContext& execution" in CUDA_CAUSAL_LM
    assert "model.execution = &execution;" in (
        CUDA_MODELS / "loader.cpp"
    ).read_text(encoding="utf-8")
    assert "struct CudaSessionCodec" in CUDA_CAUSAL_LM
    assert "CudaSessionCodec<Model>::capture" in CUDA_CAUSAL_LM_IMPL
    assert "state.decode_position_delta = decode_position_delta" in CUDA_CAUSAL_LM_IMPL
    assert "decode_position_delta = state.decode_position_delta" in CUDA_CAUSAL_LM_IMPL
    assert "void begin_speculative_suffix(int64_t draft_tokens);" in CUDA_CAUSAL_LM
    assert "CausalLm<Model>::begin_speculative_suffix" in CUDA_CAUSAL_LM_IMPL
    assert "CausalLm<Model>::finalize_hidden" in CUDA_CAUSAL_LM_IMPL
    assert "if constexpr" not in CUDA_CAUSAL_LM_IMPL
    assert "struct Qwen35Model :" not in CUDA_CAUSAL_LM
    assert "template struct CausalLm<" not in CUDA_CAUSAL_LM_SOURCE
    for model, path in (
        ("Qwen35Model", "qwen35"),
        ("MiniCPMO45Model", "minicpmo45"),
        ("MiniCPMOTtsModel", "minicpmo45"),
        ("Gemma4Model", "gemma4"),
        ("GlmDsaModel", "glm_dsa"),
        ("Glm5Model", "glm5_next"),
        ("Qwen4Model", "qwen4_exp"),
        ("DeepseekV4Model", "deepseek_v4"),
        ("DeepseekV41Model", "deepseek_v41"),
    ):
        source = (CUDA_MODELS / path / "causal_lm.cpp").read_text(
            encoding="utf-8"
        )
        assert f"template struct CausalLm<{model}>;" in source
    assert "CudaBackbone" not in CUDA_CAUSAL_LM + CUDA_CAUSAL_LM_IMPL
    assert "struct Request" not in CUDA_QWEN_BATCH_HEADER
    assert "struct QwenBatchExecutor::Impl" in CUDA_QWEN_BATCH_SOURCE
    assert len(CUDA_QWEN_BATCH_HEADER.splitlines()) < 80
    assert "Qwen35BatchStateAdapter" in CUDA_QWEN_BATCH_STATE
    assert "ContinuousBatchRequest" not in CUDA_QWEN_BATCH_STATE
    assert "continuous_batching_requests" not in CUDA_QWEN_BATCH_STATE


def test_cuda_model_finalizers_live_with_their_models() -> None:
    for name, source in CUDA_MODEL_FINALIZERS.items():
        namespace = "deepseek_v41_runtime" if name == "deepseek_v41" else name
        assert f"{namespace}::finalize_hidden" not in CUDA_CAUSAL_LM_SOURCE
        assert f"{namespace}::finalize_hidden" in source
    for implementation in (
        '"model.dsv4_hc_head"',
        '"model.deepseek_v41.final_collapse"',
        "final_mixer->pre(",
        "x.mean(2)",
    ):
        assert implementation not in CUDA_CAUSAL_LM_SOURCE
    assert "output_head = deepseek_v4::load_output_head(*execution, source);" in (
        CUDA_MODEL_FINALIZERS["deepseek_v4"]
    )


def test_cuda_runtime_composes_transport_scheduler_and_engine() -> None:
    engine_contract = (
        ROOT / "cpp_runtime" / "engine" / "include" / "engine.h"
    ).read_text(encoding="utf-8")
    scheduler = (
        ROOT / "cpp_runtime" / "scheduler" / "include" / "scheduler.h"
    ).read_text(encoding="utf-8")
    cuda_engine_header = (
        CUDA_RUNTIME / "cuda_engine.h"
    ).read_text(encoding="utf-8")
    assert "class Engine" in engine_contract
    assert "virtual ~Engine() = 0" in engine_contract
    assert "struct CudaEngine final : mfq::engine::Engine" in cuda_engine_header
    assert "CudaEngine engine;" in CUDA_ENGINE_SOURCE
    assert "std::unique_ptr<mfq::engine::ContinuousBatching>" in CUDA_ENGINE_SOURCE
    assert "make_cuda_continuous_batching(" in CUDA_ENGINE_SOURCE
    assert "qwen35::QwenBatchExecutor" not in CUDA_ENGINE_SOURCE
    assert "const mfq::engine::Engine& engine_" in scheduler
    assert "MfqInferenceEngine" not in (
        ROOT / "cpp_runtime" / "core" / "include" / "mfq" / "runtime.h"
    ).read_text(encoding="utf-8")
    assert "LoadedCudaEngine" not in CUDA_BACKEND_SOURCE
    assert "LoadedEngine" not in CUDA_BACKEND_SOURCE
    assert '"device_free_bytes"' in CUDA_ENGINE_SOURCE
    assert "MfqRuntime" not in CUDA_ENGINE_SOURCE
    assert "make_mfq_http_transport" not in CUDA_ENGINE_SOURCE
    assert "make_mfq_stdio_transport" not in CUDA_ENGINE_SOURCE
    assert "load_cuda_engine" in CUDA_RUNTIME_COMMAND
    assert "MfqRuntime runtime(" in CUDA_RUNTIME_COMMAND
    assert "make_mfq_http_transport" in CUDA_RUNTIME_COMMAND
    assert "make_mfq_stdio_transport" in CUDA_RUNTIME_COMMAND


def test_development_rules_forbid_architecture_bound_reuse() -> None:
    normalized = " ".join(CONTRIBUTING.split())
    assert "reusable code must not be architecture-bound" in normalized
    assert "mandatory extraction point" in normalized
    runtime_readme = " ".join(
        (ROOT / "cpp_runtime" / "README.md")
        .read_text(encoding="utf-8")
        .split()
    )
    assert "must never live in a model-architecture directory" in runtime_readme


def test_sparse_attention_and_deepselect_are_runtime_owned() -> None:
    assert "../models/" not in SPARSE_OPERATOR
    assert "../models/" not in DSA_OPERATOR
    assert "mlx_deepselect_topk512(" in SPARSE_HEADER
    assert "mlx_deepselect_topk512_preferred(" in SPARSE_HEADER
    assert "kDeepSelectTopkSource" in SPARSE_OPERATOR
    assert "mlx_dsa_indexer_scores(" in DSA_HEADER
    assert "mlx_cache_write_inplace(" in DSA_HEADER
    assert not (MODELS / "deepseek_v4" / "mlx_deepseek_v4_sparse.cpp").exists()
    assert not (MODELS / "deepseek_v4" / "mlx_deepseek_v4_sparse.h").exists()
    assert not (
        MODELS
        / "deepseek_v4"
        / "mlx_deepseek_v4_sparse_kernels.inc"
    ).exists()


def test_v41_never_imports_v4_runtime_implementation() -> None:
    for source in (MODELS / "deepseek_v41").glob("*.cpp"):
        text = source.read_text(encoding="utf-8")
        assert '"mlx_deepseek_v4_' not in text
    assert "mlx_yarn_tables(" in TRANSFORMER_HEADER
    assert "mlx_rope_adjacent(" in TRANSFORMER_HEADER


def test_native_mx_activation_boundaries_are_operator_owned() -> None:
    for symbol in (
        "mlx_mxfp8_sim(",
        "mlx_mxfp4_e4m3_scale_sim(",
        "mlx_weighted_rms_rope_mxfp8_sim(",
    ):
        assert symbol in MX_HEADER
    v41_attention = (
        MODELS / "deepseek_v41" / "mlx_deepseek_v41_attention.cpp"
    ).read_text(encoding="utf-8")
    assert "kMxfpSimHeader" not in v41_attention
    assert "deepseek_v41_mxfp8_e4m3_sim" not in model_sources()
    assert "mlx_apple_chip_is(\"Apple M3 Ultra\")" in MX_OPERATOR


def test_engram_streaming_reuses_the_generic_mxfp8_row_store() -> None:
    assert "MlxMxfp8RowStore" in DSV41_ENGRAM
    assert "WorkerPool" in MXFP8_ROW_STORE
    assert "DeepSeek" not in MXFP8_ROW_STORE
    assert "prefetched_rows" in DSV41_ENGRAM
    assert not (METAL / "storage" / "deepseek_v41_engram_store.cpp").exists()
    assert not (METAL / "storage" / "deepseek_v41_engram_store.h").exists()


def test_metal_device_capability_detection_is_runtime_owned() -> None:
    assert "mlx_apple_chip_name" in PLATFORM
    assert "sysctlbyname" in PLATFORM
    for source in (
        (METAL / "ops" / "mlx_nint.cpp"),
        (METAL / "ops" / "mlx_moe.cpp"),
        (METAL / "ops" / "mlx_sparse_attention.cpp"),
        (METAL / "ops" / "mlx_mx.cpp"),
        (MODELS / "minicpmo45" / "mlx_minicpmo45.cpp"),
    ):
        assert "sysctlbyname" not in source.read_text(encoding="utf-8")


def test_m3_ultra_small_m_mxfp4_dispatch_is_geometry_based() -> None:
    assert "const bool m3_ultra_native_geometry" in MOE_OPERATOR
    assert "apple_m3_ultra() && logical_experts == 256" in MOE_OPERATOR
    assert "ids[index] >= addressable_experts" in MOE_OPERATOR
    assert "input_width == 4096" in MOE_OPERATOR
    assert "output_width == 2048 || output_width == 4096" in MOE_OPERATOR
    assert "input_width == 2048 && output_width == 4096" in MOE_OPERATOR


def test_deepseek_v4_split_resident_moe_keeps_fused_pair_path() -> None:
    small_m = DSV_MOE[DSV_MOE.index("const bool smallm_gather_qmm =") :]
    small_m = small_m[: small_m.index("} else if (grouped_prefill)")]
    assert "!split_resident" in small_m
    assert "gate->forward_sorted(" not in small_m

    direct = DSV_MOE[DSV_MOE.index('"moe.dispatch.resident_direct"') :]
    direct = direct[: direct.index("if (detail::component_profile_active())")]
    assert "gate->swiglu_pair(" in direct


def test_qwen4_uses_the_shared_ssd_expert_cache() -> None:
    assert "std::optional<std::size_t> expert_cache_bytes" in QWEN4_HEADER
    assert "std::shared_ptr<MlxMoeSsdExpertCache>" in QWEN4
    assert "ssd_expert_cache_->prepare_routes(" in QWEN4
    assert "ssd_expert_cache_->prefetch_layer(" in QWEN4
    assert '"predictor.block." + std::to_string(index)' in QWEN4
    qwen4_server = DECODE_APP[DECODE_APP.index('backbone == "qwen4_exp"') :]
    assert "requested_cache_bytes(" in qwen4_server
    assert "container, context, expert_cache_bytes" in qwen4_server
    assert "Qwen4ExpertCache" not in model_sources()


def test_qwen4_small_m_down_reduce_is_format_neutral() -> None:
    moe = QWEN4[QWEN4.index("class Qwen4Moe") :]
    assert moe.count("const bool combine_routes = tokens <= 6;") == 2
    assert moe.count("routed_matmul_reduce(") >= 2
    assert "supports_mxfp4_blocks" not in moe


def test_qwen4_qsa_caches_completed_index_blocks_incrementally() -> None:
    assert "MlxSequenceCache pooled_index_cache_;" in QWEN4
    assert "const int cached = pooled_index_cache_.position();" in QWEN4
    assert "pooled_index_cache_.append(pool_index_keys(" in QWEN4
    assert "return pooled_index_cache_.view();" in QWEN4
    assert "trim_pooled_index_cache();" in QWEN4


def test_native_runtime_prewarms_shared_ssd_arenas_on_load_and_reload() -> None:
    serving = DECODE_APP[
        DECODE_APP.index("int run_loaded_runtime(") :
        DECODE_APP.index("int run_native_runtime(")
    ]
    assert "model.prewarm_ssd_expert_arena();" in serving
    # Three capability checks and their matching calls cover initial load,
    # successful context reload, and restoration after a failed reload.
    assert serving.count(".prewarm_ssd_expert_arena();") == 6


def test_dspark_moe_is_explicitly_text_only() -> None:
    dspark = (
        ROOT
        / "cpp_runtime/backends/metal/models/deepseek_v4/"
        "mlx_deepseek_v4_dspark.cpp"
    ).read_text()
    block = dspark[dspark.index("auto branches = stage.components.moe.forward_branches(") :]
    block = block[: block.index("return deepseek_v4_hc_post_sum(")]
    assert "token_ids,\n            nullptr,\n            false);" in block


def test_deepseek_v41_moe_uses_fused_down_reduce_for_every_backing() -> None:
    source = (
        ROOT
        / "cpp_runtime/backends/metal/models/deepseek_v41/"
        "mlx_deepseek_v41_moe.cpp"
    ).read_text()
    forward = source[source.index("MlxDeepseekV41Moe::forward(") :]
    assert "prepared.weights().down.combine(" in forward
    assert "down.routed_matmul_reduce(" in forward
    assert "routed_down_->combine(" in forward
    assert "moe_weighted_reduce(routed_pairs" not in forward


def test_deepseek_v41_resident_split_gate_up_keeps_two_projection_weight() -> None:
    header = (
        ROOT
        / "cpp_runtime/backends/metal/models/deepseek_v41/"
        "mlx_deepseek_v41_moe.h"
    ).read_text()
    source = DSV41_MOE
    assert "std::optional<MlxMoeWeight> routed_gate_up" in header
    assert "std::optional<MlxMoeWeight> gate_up;" in source
    assert "routed_gate_up_->routed_swiglu(" in source
    assert "std::optional<MlxRoutedLinear> routed_gate_up" not in header


def test_dspark_reuses_model_neutral_inverse_rope_output_projection() -> None:
    tensor_header = (
        METAL / "runtime" / "mlx_tensor.h"
    ).read_text(encoding="utf-8")
    tensor_source = (
        METAL / "runtime" / "mlx_tensor.cpp"
    ).read_text(encoding="utf-8")
    assert "MlxLinear::grouped_row_matmul_inverse_rope(" in tensor_source
    assert "grouped_row_matmul_inverse_rope(" in tensor_header
    for source in (DSV_DSPARK, DSV41_DSPARK):
        assert ".grouped_row_matmul_inverse_rope(" in source


def test_dspark_reads_circular_context_in_chronological_order() -> None:
    transformer_header = (
        METAL / "runtime" / "mlx_transformer.h"
    ).read_text(encoding="utf-8")
    transformer_source = (
        METAL / "runtime" / "mlx_transformer.cpp"
    ).read_text(encoding="utf-8")
    assert "mlx_circular_cache_history(" in transformer_header
    assert "array mlx_circular_cache_history(" in transformer_source
    for source in (DSV_DSPARK, DSV41_DSPARK):
        assert "mlx_circular_cache_history(ring, position)" in source
        assert "slice_axis(ring, 1, 0, active)" not in source


def test_deepseek_attention_input_projection_grouping_is_runtime_owned() -> None:
    tensor_header = (
        METAL / "runtime" / "mlx_tensor.h"
    ).read_text(encoding="utf-8")
    tensor_source = (
        METAL / "runtime" / "mlx_tensor.cpp"
    ).read_text(encoding="utf-8")
    assert "mlx_group_linears(" in tensor_header
    assert "std::optional<MlxGroupedLinear> mlx_group_linears(" in tensor_source
    assert "class MlxProjectionBatch" in tensor_header
    assert "MlxProjectionBatch::MlxProjectionBatch(" in tensor_source

    v4_attention = (
        ROOT
        / "cpp_runtime/backends/metal/models/deepseek_v4/"
        "mlx_deepseek_v4_attention.cpp"
    ).read_text(encoding="utf-8")
    assert "std::optional<MlxProjectionBatch> projections;" in v4_attention
    assert "class ProjectionGroup" not in v4_attention

    v41_header = (
        ROOT
        / "cpp_runtime/backends/metal/models/deepseek_v41/"
        "mlx_deepseek_v41_attention.h"
    ).read_text(encoding="utf-8")
    v41_attention = (
        ROOT
        / "cpp_runtime/backends/metal/models/deepseek_v41/"
        "mlx_deepseek_v41_attention.cpp"
    ).read_text(encoding="utf-8")
    assert "std::optional<MlxProjectionBatch> input_projections_;" in v41_header
    assert "input_projections_.emplace(std::move(input_projections));" in v41_attention
    assert "(*input_projections_)(input)" in v41_attention

    for source in (DSV_DSPARK, DSV41_DSPARK):
        assert "MlxProjectionBatch input_projections;" in source
        assert "input_projections(std::vector<const MlxLinear*>" in source
        assert "stage.input_projections(input)" in source


def test_deepseek_v4_mfe_streaming_uses_fused_down_reduce() -> None:
    source = (
        ROOT
        / "cpp_runtime/backends/metal/models/deepseek_v4/"
        "mlx_deepseek_v4_moe.cpp"
    ).read_text()
    streamed = source[source.index("} else if (!expert_offload_) {") :]
    streamed = streamed[: streamed.index("if (!shared.has_value())")]
    assert "down_weight.routed_matmul_reduce(" in streamed
    assert "return moe_weighted_reduce(" not in streamed


def test_deepseek_v4_only_tracks_token_counts_for_active_penalties() -> None:
    source = (
        ROOT
        / "cpp_runtime/backends/metal/models/deepseek_v4/"
        "mlx_deepseek_v4_causal_lm.cpp"
    ).read_text()
    generation = source[source.index("MlxDeepseekV4CausalLm::generate_impl(") :]
    assert "std::optional<array> counts;" in generation
    assert "if (sampling.has_penalties())" in generation
    assert "? sampler.sample(logits, *counts)" in generation
    assert "if (counts) {\n            *counts = sample_token_counts_add(" in generation


def test_deepseek_v4_small_m_reuses_ssd_route_transactions() -> None:
    source = (
        ROOT
        / "cpp_runtime/backends/metal/models/deepseek_v4/"
        "mlx_deepseek_v4_causal_lm.cpp"
    ).read_text()
    forward = source[
        source.index("MlxDeepseekV4CausalLm::forward_streaming_layers(") :
        source.index("MlxDeepseekV4CausalLm::begin_speculative_target(")
    ]
    transaction_begin = forward.index("const bool route_transaction")
    transaction = forward[transaction_begin:]
    assert "&& routed_rows >= 1" in forward
    assert "&& routed_rows <= 6" in forward
    assert "has_full_residency_capacity" not in transaction
    assert "targets == nullptr" not in transaction
    assert "capture_dspark_target(group_begin, hidden, targets);" in transaction
    assert "auto trial_targets = targets != nullptr" in transaction
    assert "*targets = std::move(trial_targets);" in transaction


def test_dspark_small_m_reuses_the_shared_ssd_route_transaction() -> None:
    draft = DSV_DSPARK[DSV_DSPARK.index("MlxDeepseekV4DSpark::draft_impl(") :]
    assert "routed_rows <= 6" in draft
    assert "mlx_ssd_route_transactions_enabled()" in draft
    assert "begin_route_transaction();" in draft
    assert "resolve_route_transaction();" in draft
    assert "route_layer_likely_hit(" in draft


def test_deepseek_v41_multitoken_attention_orders_circular_cache_write() -> None:
    source = (
        ROOT
        / "cpp_runtime/backends/metal/models/deepseek_v41/"
        "mlx_deepseek_v41_attention.cpp"
    ).read_text()
    forward = source[source.index("MlxDeepseekV41Attention::forward(") :]
    assert "std::optional<array> pending_local_values;" in forward
    assert "std::optional<array> pending_local_rows;" in forward
    assert "mlx::core::depends(\n            std::vector<array>{state.local_kv},\n            std::vector<array>{attended})" in forward
    assert "ordered_local.front(),\n            *pending_local_values" in forward


def test_deepseek_v41_route_transactions_keep_wide_global_expert_ids() -> None:
    assert "snapshot.defer_transaction_global(routes.ids);" in DSV41_MOE
    assert "snapshot.defer_transaction(routes.ids);" not in DSV41_MOE


def test_direct_native_hf_server_uses_size_aware_expert_residency() -> None:
    assert "native_hf_source_bytes(" in DECODE_APP
    assert "automatic_runtime_memory_budget_bytes()" in DECODE_APP
    assert 'info.find("max_recommended_working_set_size")' in DECODE_APP
    assert "statistics.free_count" in DECODE_APP
    assert "statistics.inactive_count" in DECODE_APP
    assert "is_independently_streamed_record" in DECODE_APP
    assert "resident_source_bytes <= available_bytes" in DECODE_APP
    assert (
        "requested_cache_bytes(\n"
        "                arguments.expert_cache_gb, native_hf, container"
    ) in DECODE_APP

    assert "F_NOCACHE" in HF_SOURCE

    cache = (
        ROOT
        / "cpp_runtime/backends/metal/runtime/mlx_ssd_expert_cache.cpp"
    ).read_text(encoding="utf-8")
    assert "store.total_num_experts()" in cache


def test_mixed_mfe_offload_uses_the_shared_pager() -> None:
    for source in (QWEN4, DSV41_MOE):
        assert "can_group_mfe(" in source
        assert "grouped_mfe(" in source
    assert "MlxMfeOffloadCache" in DSV41
    assert "class MlxMfeOffloadCache" not in model_sources()


def test_mtp_policy_and_lifecycle_are_runtime_owned() -> None:
    for symbol in (
        "struct MlxMtpGenerationStats",
        "class MlxMtpDepthController",
        "struct MlxMtpEngineCallbacks",
        "run_mlx_mtp_generation",
        "verify_stochastic_mtp_top_k_chain_device",
    ):
        assert symbol in MTP_HEADER or symbol in MTP_SOURCE

    sources = model_sources()
    assert "struct MlxMtpGenerationStats" not in sources
    assert "class MlxMtpDepthController" not in sources
    assert "verify_greedy_mtp(" not in sources
    assert "verify_stochastic_mtp(" not in sources
    assert "host_sampling_distribution(" not in sources


def test_every_mtp_architecture_is_a_thin_client_of_one_engine() -> None:
    clients = (QWEN, QWEN4, DSV, DSV41)
    for source in clients:
        assert source.count("run_mlx_mtp_generation(") == 1
        assert source.count("mlx_mtp_verification_ids(") == 1
        assert source.count("mlx_mtp_committed_hidden(") == 1

    all_sources = model_sources()
    assert all_sources.count("run_mlx_mtp_generation(") == len(clients)
    assert "MlxMtpDepthPolicy" not in all_sources
    assert "AcceptanceOnly" not in all_sources
    assert "plain_decode" not in MTP_HEADER
    assert "should_exit" not in MTP_HEADER
    assert "should_exit" not in MTP_SOURCE
    assert "draft_greedy(" not in DSV
    assert "MlxMtpGenerationStats last_mtp_stats_" in (
        MODELS / "deepseek_v4" / "mlx_deepseek_v4_causal_lm.h"
    ).read_text(encoding="utf-8")
    assert "begin_speculative_target(1, draft_count + 1)" in DSV
    assert "rollback_speculative_target(" in DSV
    assert "forward_chunk(\n                    committed_ids" not in DSV


def test_dspark_adapters_keep_only_predictor_math_and_cache_state() -> None:
    for source in (DSV_DSPARK, DSV41_DSPARK):
        assert "mlx_sparse_selected_mla_attention(" in source
        assert "void MlxDeepseek" in source and "::propose(" in source
        assert "draft_impl(" in source
        assert "const int physical_width = available_width;" in source
        assert "evaluate the complete physical block" in source
    assert "stable_dspark_state_" in DSV
    assert "stable_dspark_state_" in DSV41
    assert "MlxMtpDepthPolicy" not in DSV
    assert "MlxMtpDepthPolicy" not in DSV41


def test_v41_keeps_shared_moe_hc_fusion_in_target_and_predictor() -> None:
    assert "ffn_mhc_.expand_sum(" in DSV41
    assert "stage.ffn_mhc.expand_sum(" in DSV41_DSPARK
    assert "return mlx::core::reshape(routed + shared" not in DSV41_MOE


def test_server_exposes_predictors_by_role_not_architecture_name() -> None:
    assert SERVER_COMPONENTS.count(
        'component_declared(graph, "predictor")'
    ) == 4
    assert re.search(
        r'component_with_implementation\([^;]*"predictor"',
        SERVER_COMPONENTS,
        re.DOTALL,
    ) is None


def test_model_config_parsing_is_backend_neutral() -> None:
    shared_configs = (
        "model_config",
        "minicpmo45",
        "glm5_next",
        "qwen4_exp",
        "qwen35",
        "glm_dsa",
        "gemma4",
        "deepseek_v4",
        "deepseek_v41",
    )

    assert "../models/" not in (CORE / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "target_link_libraries(mfq-models PUBLIC mfq-core)" in (
        CORE.parent / "models" / "CMakeLists.txt"
    ).read_text(encoding="utf-8")
    assert not (CUDA_MODELS / "config.h").exists()
    assert not (CUDA_RUNTIME / "model_config.h").exists()
    assert not (CUDA_MODELS / "cuda_model_config.h").exists()
    assert not (CUDA_MODELS / "cuda_model_config.cpp").exists()
    for stem in shared_configs:
        assert (CORE.parent / "models" / "include" / f"{stem}.h").is_file()
        assert (CORE.parent / "models" / f"{stem}.cpp").is_file()

    for config in (
        "mfq::models::qwen35::Config",
        "mfq::models::minicpmo45::Config",
        "mfq::models::ModelConfig",
        "mfq::models::gemma4::Config",
        "mfq::models::glm5_next::Config",
        "mfq::models::qwen4_exp::Config",
        "mfq::models::glm_dsa::Config",
        "mfq::models::deepseek_v4::Config",
        "mfq::models::deepseek_v41::Config",
    ):
        assert f"{config} config;" in CUDA_MODEL_HEADERS
        assert f"{config} config;" not in CUDA_CAUSAL_LM
    assert "CudaRuntimeParameters" not in CUDA_BACKEND_SOURCE
    assert "load_runtime_parameters" not in CUDA_BACKEND_SOURCE
    assert "resolved_config_json" not in CUDA_BACKEND_SOURCE
    assert "nlohmann::json" not in CUDA_REGISTRY + CUDA_CAUSAL_LM_LOADER
    assert "Config::from_json" not in CUDA_REGISTRY

    qwen_config = (CORE.parent / "models" / "include" / "qwen35.h").read_text(
        encoding="utf-8"
    )
    for field in (
        "attention_output_gate",
        "mtp_use_dedicated_embeddings",
        "linear_conv_kernel_dim",
        "linear_key_head_dim",
        "linear_value_head_dim",
        "linear_num_key_heads",
        "linear_num_value_heads",
        "mrope_sections",
        "mrope_interleaved",
        "grid_vision",
        "image_token_id",
        "video_token_id",
    ):
        assert field in qwen_config

    assert "Config::from_json" not in CUDA_CAUSAL_LM_LOADER
    assert "if constexpr" not in CUDA_CAUSAL_LM_LOADER
    assert "if constexpr" not in (
        CUDA_RUNTIME / "components.cpp"
    ).read_text(encoding="utf-8")
    model_names = (
        "minicpmo45",
        "glm5_next",
        "qwen4_exp",
        "deepseek_v41",
        "qwen35",
        "glm_dsa",
        "gemma4",
        "deepseek_v4",
    )
    model_loaders = "\n".join(
        (CUDA_MODELS / name / "causal_lm.cpp").read_text(encoding="utf-8")
        for name in model_names
    )
    assert model_loaders.count("Config::from_json") == 9
    for namespace in model_names:
        assert not (CUDA_MODELS / namespace / f"{namespace}_model.h").exists()
        assert not (CUDA_MODELS / namespace / f"{namespace}_model.cpp").exists()


def test_cuda_qwen4_and_glm5_own_their_model_implementations() -> None:
    assert not (CUDA_MODELS / "flash_next").exists()
    qwen = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (CUDA_MODELS / "qwen4_exp").glob("*")
        if path.is_file()
    )
    glm = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (CUDA_MODELS / "glm5_next").glob("*")
        if path.is_file()
    )
    assert "namespace mfq::cuda::qwen4_exp" in qwen
    assert "namespace mfq::cuda::glm5_next" in glm
    assert "glm5_next" not in qwen
    assert "qwen4_exp" not in glm


def test_cuda_model_runtime_uses_compiled_causal_lm_adapters() -> None:
    cmake = (ROOT / "cpp_runtime" / "backends" / "cuda" / "CMakeLists.txt").read_text(
        encoding="utf-8"
    )
    adapters = (
        "qwen4_exp",
        "glm5_next",
        "deepseek_v41",
        "deepseek_v4",
        "glm_dsa",
        "gemma4",
        "qwen35",
    )

    assert not list(CUDA_MODELS.rglob("construction.h"))
    assert not list(CUDA_MODELS.rglob("model_loader.h"))
    assert not list((ROOT / "cpp_runtime" / "backends" / "cuda").rglob("*.inc"))
    for namespace in adapters:
        model_dir = CUDA_MODELS / namespace
        header = model_dir / "causal_lm.h"
        source = model_dir / "causal_lm.cpp"
        assert header.is_file()
        assert source.is_file()
        assert f"models/{namespace}/causal_lm.cpp" in cmake
        assert '#include "causal_lm.h"' in source.read_text(encoding="utf-8")

    for concrete_definition in (
        "struct Glm5NextBlock",
        "struct Qwen4Block",
        "struct Dsv4Block",
        "struct GlmDsaBlock",
        "struct QuantLinear",
        "class MoeExpertCache",
    ):
        assert concrete_definition not in CUDA_RUNTIME_SOURCE
    causal_lm = (CUDA_MODELS / "causal_lm.h").read_text(encoding="utf-8")
    causal_lm_loader = (CUDA_MODELS / "loader.cpp").read_text(
        encoding="utf-8"
    )
    assert re.search(r"\bCudaModel\b", CUDA_BACKEND_SOURCE) is None
    assert "switch (m.c.runtime_plan.backbone)" not in causal_lm_loader
    for runtime in (
        "Qwen35CausalLm",
        "MiniCPMO45CausalLm",
        "Gemma4CausalLm",
        "GlmDsaCausalLm",
        "Glm5CausalLm",
        "Qwen4CausalLm",
        "DeepseekV4CausalLm",
        "DeepseekV41CausalLm",
    ):
        assert f"using {runtime} = CausalLm<" in causal_lm

    assert "std::make_unique<FullBlock>" in CUDA_TRANSFORMER_LOADER
    assert "std::make_unique<LinearAttentionBlock>" in CUDA_QWEN_LINEAR
    assert 'type == "linear_attention"' not in CUDA_TRANSFORMER_LOADER
    assert len(CUDA_RUNTIME_SOURCE.splitlines()) < 12_000


def test_cuda_ops_and_execution_are_real_compilation_units() -> None:
    cuda = ROOT / "cpp_runtime" / "backends" / "cuda"
    cmake = (ROOT / "cpp_runtime" / "backends" / "cuda" / "CMakeLists.txt").read_text(
        encoding="utf-8"
    )
    required = (
        "ops/format.cpp",
        "ops/fp8_sq.cpp",
        "ops/moe.cpp",
        "ops/mx.cpp",
        "ops/mxfp4_sq.cpp",
        "ops/nint.cpp",
        "ops/quant_linear.cpp",
        "ops/vq.cpp",
        "ops/cuda_execution.cpp",
        "engine/decode_graph.cpp",
        "engine/options.cpp",
        "commands/cli.cpp",
        "models/causal_lm.cpp",
        "models/loader.cpp",
        "models/transformer.cpp",
        "engine/mtp.cpp",
        "storage/moe_expert_cache.cpp",
        "engine/components.cpp",
        "engine/cuda_batching.cpp",
        "models/qwen35/batch_state.cpp",
        "diagnostics/attention_checks.cpp",
        "diagnostics/backend_checks.cpp",
        "diagnostics/linear_checks.cpp",
        "diagnostics/moe_checks.cpp",
        "diagnostics/session_checks.cpp",
        "eval/kl.cpp",
    )
    for relative in required:
        assert (cuda / relative).is_file()
        assert relative in cmake
    assert not (cuda / "engine" / "runner.h").exists()
    assert not (cuda / "engine" / "runner.cpp").exists()
    ops = cuda / "ops"
    assert "${MFQ_CUDA_ROOT}/ops/include" in cmake
    assert not any(path.suffix == ".h" for path in ops.iterdir())
    quant_header = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ops / "include").glob("quant_linear*.h")
    )
    assert "struct QuantLinear" in quant_header
    owned_types = {
        "ops/include/nint.h": ("NintWeight",),
        "ops/include/vq.h": ("NvqWeight",),
        "ops/include/mx.h": ("Mxfp4Weight", "Mxfp8Weight"),
        "ops/include/fp8_sq.h": ("Fp8SqWeight",),
        "ops/include/mxfp4_sq.h": ("Mxfp4SqWeight",),
        "ops/include/mfe_weight.h": ("MfeWeight",),
        "ops/include/moe_types.h": ("MoeRoutePlan",),
    }
    for relative, names in owned_types.items():
        source = (cuda / relative).read_text(encoding="utf-8")
        for name in names:
            assert f"struct {name}" in source
            assert f"struct {name} {{" not in quant_header
    assert "struct QuantLinear" not in CUDA_RUNTIME_SOURCE
    assert "cuda_quantized_ops" not in CUDA_BACKEND_SOURCE
    assert not re.search(r'#include\s+["<][^">]+\.inc[">]', CUDA_BACKEND_SOURCE)


def test_cuda_qwen_speculation_is_model_owned() -> None:
    assert "struct LinearAttentionBlock final" in CUDA_QWEN_LINEAR
    assert "LinearAttentionBlock::rollback_speculative" in CUDA_QWEN_LINEAR
    assert "replay_recurrent_cuda(" in CUDA_QWEN_LINEAR
    assert "forward_speculative(" not in CUDA_TRANSFORMER_HEADER
    assert "struct LinearBlock" not in CUDA_TRANSFORMER_HEADER
    assert "MFQ_QWEN_MTP_BATCH" not in CUDA_TRANSFORMER_HEADER
    assert "for (int accepted_drafts : {0, 1, 2})" in CUDA_BACKEND_SOURCE


def test_cuda_qwen_linear_ffn_matches_residual_dtype() -> None:
    start = CUDA_QWEN_LINEAR.index("linear.ffn_residual")
    residual = CUDA_QWEN_LINEAR[start : start + 800]
    assert "ff2.scalar_type() != rr.scalar_type()" in residual
    assert "ff2 = ff2.to(rr.scalar_type()).contiguous();" in residual


def test_cuda_mtp_generation_loop_is_architecture_independent_and_reversible() -> None:
    generation = CUDA_MTP_SOURCE
    implementation = generation[: generation.index("#define MFQ_INSTANTIATE_MTP")]
    assert "run_mtp_generation(" in implementation
    assert "int32_t run_mtp_generation(" not in CUDA_RUNTIME_SOURCE
    for architecture_name in (
        "Qwen",
        "DeepSeek",
        "GLM",
        "Flash",
        "MiniCPM",
    ):
        assert architecture_name not in implementation
    assert "should_exit" not in generation
    assert "should_exit" not in CUDA_MTP_HEADER
    assert "exit_streak" not in CUDA_MTP_HEADER
    assert "DepthController depth_controller" in generation
    assert "bounded_depth(depth_controller.depth())" in generation
    assert "retains_partial_target_prefix()" in generation
    assert "CompactDistribution" in generation
    assert "mfq_tensor_backend::topk(" in generation


def test_generic_generation_and_sequence_cache_helpers_are_not_redeclared() -> None:
    assert "mlx_last_token_logits" in SAMPLING_HEADER
    assert "class MlxSequenceCache" in TRANSFORMER_HEADER
    sources = model_sources()
    forbidden_definitions = (
        r"\b(?:array|mlx::core::array)\s+last_(?:token_)?logits\s*\(",
        r"\bclass\s+SequenceCache\b",
        r"\bstd::optional<[^>]*array[^>]*>\s+\w*generation_token_counts\s*\(",
    )
    for pattern in forbidden_definitions:
        assert re.search(pattern, sources) is None, pattern
