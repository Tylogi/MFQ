"""CPU primitives and file I/O must not pull in expert execution or the old core."""
from pathlib import Path
import json
import re
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from include_graph import sources
from test_cmake_backend_switch import CSRC, NO_CUDA_TOOLKIT, NOT_APPLE, _configure
from test_cuda_kernel_torch_free import closure
from mfq.commands.build import _cuda_compiler


def _target_dependencies(graph: str, target: str) -> set[str]:
    nodes = {
        node: label.split("\\n")[0]
        for node, label in re.findall(r'"([^"]+)"\s*\[\s*label\s*=\s*"([^"]+)"', graph)
    }
    roots = [node for node, name in nodes.items() if name == target]
    assert len(roots) == 1, f"missing or ambiguous target {target}"
    edges = re.findall(r'"([^"]+)"\s*->\s*"([^"]+)"', graph)
    seen, pending = set(), roots
    while pending:
        node = pending.pop()
        if node in seen:
            continue
        seen.add(node)
        pending.extend(child for parent, child in edges if parent == node)
    return {nodes[node] for node in seen}


def _check_native_targets(graph: str, cuda: bool) -> None:
    forbidden = {"mfq-core", "mfq-model-source", "mfq-model-graph", "mfq-json", "mfq-cpu-execution",
                 "mfq-engine", "mfq-models"}
    targets = ["mfq-host-parallel", "mfq-host-parallel-test", "mfq-cpu-topology-test",
               "mfq-cpu-ops", "mfq-host-memory-copy-test", "mfq-io", "mfq-file-range-reader-test"]
    if cuda:
        targets.append("mfq-cuda-native")
    for target in targets:
        dependencies = _target_dependencies(graph, target)
        assert "mfq-host-parallel" in dependencies
        assert not dependencies & forbidden, f"{target} reaches {dependencies & forbidden}"
    assert "mfq-cpu-execution" in _target_dependencies(graph, "mfq-cpu-expert-pool-test")
    assert "mfq-host-parallel" in _target_dependencies(graph, "mfq-cpu-execution")
    if cuda:
        _check_kernel_target(graph)


def _check_kernel_target(graph: str) -> None:
    dependencies = _target_dependencies(graph, "mfq-cuda-native-kernels")
    assert "mfq-formats" in dependencies
    forbidden = {"mfq-core", "mfq-model-source", "mfq-model-graph", "mfq-engine", "mfq-models",
                 "mfq-json", "mfq-io", "mfq-quant-readers", "mfq-cpu-ops", "mfq-cpu-execution", "mfq-qsa-kv-store"}
    assert not dependencies & forbidden, f"kernels reach {dependencies & forbidden}"
    assert _target_dependencies(graph, "mfq-formats") == {"mfq-formats"}


def test_kernel_target_check_rejects_transitive_core_dependency() -> None:
    graph = '\n'.join(f'"n{i}" [ label = "{name}" ];' for i, name in enumerate(
        ["mfq-cuda-native-kernels", "mfq-formats", "bridge", "mfq-core"]))
    graph += '\n"n0" -> "n1";\n"n0" -> "n2";'
    _check_kernel_target(graph)
    with pytest.raises(AssertionError, match="kernels reach"):
        _check_kernel_target(graph + '\n"n2" -> "n3";')


def _check_kernel_sources(csrc: Path) -> None:
    core = csrc / "core"
    units = list(sources(str(csrc / "backends/cuda/kernels")))
    assert units, "CUDA kernel sources missing"
    for unit in units:
        reached, _ = closure(unit, str(csrc))
        escaped = [path for path in reached if Path(path).is_relative_to(core)]
        assert not escaped, f"{unit} reaches core sources: {escaped}"


def test_kernel_source_closure() -> None:
    _check_kernel_sources(CSRC)


def test_kernel_source_check_rejects_transitive_core_include(tmp_path: Path) -> None:
    kernel = tmp_path / "backends/cuda/kernels"
    kernel.mkdir(parents=True)
    (kernel / "read.cu").write_text('#include "bridge.h"\n')
    bridge = kernel / "bridge.h"
    bridge.write_text('#include "quant/blob.h"\n')
    blob = tmp_path / "formats/quant/blob.h"
    blob.parent.mkdir(parents=True)
    blob.write_text('#pragma once\n')
    _check_kernel_sources(tmp_path)
    forbidden = tmp_path / "core/include/mfq/runtime.h"
    forbidden.parent.mkdir(parents=True)
    forbidden.write_text('#pragma once\n')
    blob.write_text('#include <mfq/runtime.h>\n')
    with pytest.raises(AssertionError, match="reaches core sources"):
        _check_kernel_sources(tmp_path)


@pytest.mark.parametrize("cuda", [False, True], ids=["host-only", "cuda"])
@pytest.mark.parametrize("communication", [False, True], ids=["no-communication", "communication"])
def test_native_target_closure(tmp_path: Path, cuda: bool, communication: bool) -> None:
    if cuda and (not NOT_APPLE or _cuda_compiler() is None):
        pytest.skip("CUDA target graph requires a supported CUDA toolchain")
    options = [f"-DMFQ_BUILD_CUDA_RUNTIME={'ON' if cuda else 'OFF'}", "-DMFQ_BUILD_METAL_RUNTIME=OFF",
               f"-DMFQ_BUILD_RUNTIME_COMMUNICATION={'ON' if communication else 'OFF'}",
               "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"]
    if not cuda:
        options.extend(NO_CUDA_TOOLKIT)
    code, output, _ = _configure(tmp_path, "native-closure", *options)
    assert code == 0, output
    _check_native_targets((tmp_path / "native-closure/targets.dot").read_text(), cuda)
    readers = _target_dependencies((tmp_path / "native-closure/targets.dot").read_text(),
                                   "mfq-quant-readers")
    assert {"mfq-formats", "mfq-io", "mfq-host-parallel"} <= readers
    assert not readers & {"mfq-core", "mfq-model-source", "mfq-model-graph", "mfq-json",
                          "mfq-engine", "mfq-models", "mfq-cpu-execution"}
    graph = (tmp_path / "native-closure/targets.dot").read_text()
    for target in ("mfq-model-graph", "mfq-model-source"):
        deps = _target_dependencies(graph, target)
        assert not deps & {"mfq-core", "mfq-engine", "mfq-models", "mfq-cpu-execution",
                           "mfq-quant-readers", "mfq-cuda-native"}
    assert "mfq-model-source" not in _target_dependencies(graph, "mfq-model-graph")
    commands = json.loads((tmp_path / "native-closure/compile_commands.json").read_text())
    inputs = [item for item in commands if any(target in item["command"] for target in
              ("mfq-model-source.dir", "mfq-model-graph.dir"))]
    assert inputs
    assert all((CSRC / "core").as_posix() not in item["command"] for item in inputs)
    clients = [item for item in commands if any(target in item["command"] for target in
               ("mfq-engine-contract-test.dir", "mfq-runtime-scheduler.dir"))]
    assert clients
    for item in clients:
        assert (CSRC / "engine/include").as_posix() in item["command"]
        assert (CSRC / "core").as_posix() not in item["command"]
        assert (CSRC / "engine/model").as_posix() not in item["command"]
    schema = [item for item in inputs if "mfq-model-graph.dir" in item["command"]]
    assert len(schema) == 1 and schema[0]["file"].endswith("formats/model_graph/model_graph.cpp")
    if cuda:
        commands = json.loads((tmp_path / "native-closure/compile_commands.json").read_text())
        kernels = [item["command"].replace("\\", "/") for item in commands
                   if "mfq-cuda-native-kernels.dir" in item["command"]]
        assert kernels, "native kernel compile commands missing"
        assert all((CSRC / "formats").as_posix() in command for command in kernels)
        assert all((CSRC / "core").as_posix() not in command for command in kernels)


def test_target_check_rejects_transitive_expert_dependency() -> None:
    targets = ["mfq-host-parallel", "mfq-host-parallel-test", "mfq-cpu-topology-test",
               "mfq-cpu-execution", "mfq-cpu-expert-pool-test", "bridge", "mfq-cpu-ops",
               "mfq-host-memory-copy-test", "mfq-io", "mfq-file-range-reader-test"]
    graph = "\n".join(f'"n{i}" [ label = "{name}" ];' for i, name in enumerate(targets))
    graph += '\n"n1" -> "n0";\n"n2" -> "n0";\n"n3" -> "n0";\n"n4" -> "n3";'
    graph += '\n"n6" -> "n0";\n"n7" -> "n6";\n"n8" -> "n0";\n"n9" -> "n8";'
    _check_native_targets(graph, False)
    with pytest.raises(AssertionError, match="mfq-host-parallel reaches"):
        _check_native_targets(graph + '\n"n0" -> "n5";\n"n5" -> "n3";', False)
    with pytest.raises(AssertionError, match="mfq-io reaches"):
        _check_native_targets(graph + '\n"n8" -> "n5";\n"n5" -> "n3";', False)


def _check_cuda_native_sources(csrc: Path) -> None:
    native = csrc / "backends/cuda/native"
    units = list(sources(str(native)))
    assert units, "CUDA native sources missing"
    for unit in units:
        reached, _ = closure(unit, str(csrc))
        escaped = [path for path in reached if not (
            Path(path).is_relative_to(native) or
            Path(path).is_relative_to(csrc / "backends/cpu/native") or
            Path(path) == csrc / "engine/include/resource_error.h")]
        assert not escaped, f"{unit} reaches non-native sources: {escaped}"


def test_cuda_native_contract_closure() -> None:
    _check_cuda_native_sources(CSRC)


def test_cuda_native_check_rejects_request_include(tmp_path: Path) -> None:
    native = tmp_path / "backends/cuda/native"
    native.mkdir(parents=True)
    (native / "context.cu").write_text('#include "resource_error.h"\n')
    public = tmp_path / "engine/include"
    public.mkdir(parents=True)
    error = public / "resource_error.h"
    error.write_text("#pragma once\n")
    _check_cuda_native_sources(tmp_path)
    (public / "request.h").write_text("struct Request {};\n")
    error.write_text('#include "request.h"\n')
    with pytest.raises(AssertionError, match="reaches non-native sources"):
        _check_cuda_native_sources(tmp_path)


def _check_native_sources(csrc: Path) -> None:
    native = csrc / "backends/cpu/native"
    units = list(sources(str(native)))
    assert units, "CPU native sources missing"
    for unit in units:
        reached, torch_headers = closure(unit, str(csrc))
        escaped = [path for path in reached if not Path(path).is_relative_to(native)]
        assert not escaped, f"{unit} reaches non-native sources: {escaped}"
        assert not torch_headers, f"{unit} reaches Torch headers: {torch_headers}"


def test_native_source_closure() -> None:
    _check_native_sources(CSRC)


@pytest.mark.parametrize("target", [
    "model_source/model_source.h",
    "engine/include/engine.h",
    "components/json/json.h",
    "backends/cpu/execution/include/mfq/moe/cpu_expert_pool.h",
])
def test_source_check_rejects_transitive_non_native_include(tmp_path: Path, target: str) -> None:
    native = tmp_path / "backends/cpu/native"
    native.mkdir(parents=True)
    (native / "host.cpp").write_text('#include "bridge.h"\n')
    bridge = native / "bridge.h"
    bridge.write_text("#include <cstdint>\n")
    _check_native_sources(tmp_path)
    forbidden = tmp_path / target
    forbidden.parent.mkdir(parents=True)
    forbidden.write_text("#pragma once\n")
    bridge.write_text(f"#include<{target}>\n")
    with pytest.raises(AssertionError, match="reaches non-native sources"):
        _check_native_sources(tmp_path)
