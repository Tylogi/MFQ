"""CPU primitives and file I/O must not pull in expert execution or the old core."""
from pathlib import Path
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


@pytest.mark.parametrize("cuda", [False, True], ids=["host-only", "cuda"])
def test_native_target_closure(tmp_path: Path, cuda: bool) -> None:
    if cuda and (not NOT_APPLE or _cuda_compiler() is None):
        pytest.skip("CUDA target graph requires a supported CUDA toolchain")
    options = [f"-DMFQ_BUILD_CUDA_RUNTIME={'ON' if cuda else 'OFF'}", "-DMFQ_BUILD_METAL_RUNTIME=OFF"]
    if not cuda:
        options.extend(NO_CUDA_TOOLKIT)
    code, output, _ = _configure(tmp_path, "native-closure", *options)
    assert code == 0, output
    _check_native_targets((tmp_path / "native-closure/targets.dot").read_text(), cuda)


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
    "core/include/mfq/model_source/model_source.h",
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
