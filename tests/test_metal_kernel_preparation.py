from pathlib import Path


METAL = Path(__file__).resolve().parents[1] / "cpp_runtime/backends/metal"


def test_preparation_does_not_execute_graphs():
    source = (METAL / "runtime/mlx_kernel_prepare.cpp").read_text()
    for operation in (".eval(", "mlx::core::eval(", "get_command_encoder(",
                      "dispatch_threads(", "dispatch_threadgroups(", "allocator::malloc("):
        assert operation not in source
    assert "device.get_library(" in source
    assert "device.get_kernel(" in source


def test_custom_kernel_cache_matches_source_and_compile_options():
    source = (METAL / "runtime/mlx_kernel_prepare.cpp").read_text()
    assert "MLX_VERSION_NUMERIC >= 32001" in source
    assert "std::hash<std::string>{}(std::get<1>(state))" in source
    assert "std::get<10>(state)" in source
    assert "device.get_library(library_name," in source


def test_preparation_covers_host_and_device_sampling_inputs():
    source = (METAL / "runtime/mlx_kernel_prepare.cpp").read_text()
    assert "mlx::core::zeros({rows}, mlx::core::float32)" in source
    assert "mlx::core::array(random_values.begin(), mlx::core::Shape{rows})" in source
    assert "mlx::core::bfloat16" in source


def test_load_prepares_kernels_before_service_ready():
    source = (METAL / "apps/mfq_decode_mlx.cpp").read_text()
    loaded = source.split("int run_loaded_runtime(", 1)[1].split("int run_native_runtime(", 1)[0]
    assert loaded.index("preparation->finish();") < loaded.index("return runtime.run();")
    assert "preparation.collect_sampling(preparation_vocab, preparation_sampling);" in source


def test_warmup_is_internal_and_cleans_state_before_ready():
    source = (METAL / "apps/mfq_decode_mlx.cpp").read_text()
    warmup = source.split("void warm_runtime()", 1)[1].split("std::shared_ptr<std::mutex> runtime_mutex", 1)[0]
    assert "generate_with_prefill_metrics(" in warmup
    assert "session_cache->" not in warmup
    assert "runtime.reset_generation_state();" in warmup
    assert "release_model_load_staging_memory(runtime_stream);" in warmup
    for family, filename in (("flash_next", "mlx_qwen4_causal_lm.cpp"), ("qwen35", "mlx_qwen35_causal_lm.cpp")):
        model = (METAL / "models" / family / filename).read_text()
        reset = model.split("::reset_generation_state() noexcept", 1)[1].split("\n}", 1)[0]
        assert "clear_cache();" in reset and "policy_state" in reset and "last_mtp_stats" in reset
