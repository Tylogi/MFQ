from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CUDA_ROOT = ROOT / "cpp_runtime" / "backends" / "cuda"
SOURCE = "\n".join(
    path.read_text(encoding="utf-8")
    for path in sorted(CUDA_ROOT.rglob("*"))
    if path.suffix in {".h", ".cpp"}
)


def test_decode_branch_executor_forks_and_joins_cuda_streams() -> None:
    assert "struct CudaIndependentBranchExecutor" in SOURCE
    assert "cudaEventRecord(" in SOURCE
    assert "cudaStreamWaitEvent(" in SOURCE
    assert "mfq_cuda_record_stream(" in SOURCE


def test_decode_branch_parallelism_is_decode_only_and_graph_safe() -> None:
    assert "return rows == 1" in SOURCE
    assert "cudaStreamIsCapturing(" in SOURCE
    assert "capture_status != cudaStreamCaptureStatusNone" in SOURCE


def test_runtime_parallelizes_important_neuron_branches_only_for_decode() -> None:
    source = (CUDA_ROOT / "core/ffn.cpp").read_text(encoding="utf-8")
    branch = source.split("if (allow_important_neurons && important_neurons)", 1)[1]
    branch = branch.split("if (tensor_parallel_dense_compatible())", 1)[0]
    assert "decode_branch_parallel_enabled(execution.config,execution.decode_graph_serial_branches,rows)&&" in "".join(branch.split())


def test_incompatible_projection_groups_use_the_common_executor() -> None:
    assert "split_w.size()," in SOURCE
    assert "layers.size() - 1," in SOURCE
    assert "layers.size()," in SOURCE


def test_mixed_precision_group_diagnostic_captures_and_replays_cuda_graph() -> None:
    assert "linear_group_graph_check" in SOURCE
    assert "graph.capture_begin();" in SOURCE
    assert "graph.capture_end();" in SOURCE
    assert "graph.replay();" in SOURCE
