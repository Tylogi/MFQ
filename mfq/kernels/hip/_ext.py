"""Loader for the unified mfq_hip extension on AMD ROCm hosts.

Mirrors mfq.kernels.cuda._ext: one extension carries every GPU op
(norm/acc/rope/attention/gated-delta-net/kv-cache/embedding/sampling/nint and
nvq matmul), built from the hipify-translated kernels in this directory.
Routed to automatically by mfq.kernels.cuda._ext.ext() when torch reports a
HIP build.
"""

from __future__ import annotations

import os
from pathlib import Path

_DIR = os.path.dirname(__file__)
_REPOSITORY_ROOT = str(Path(_DIR).resolve().parents[2])
_HIP_RUNTIME_INCLUDE = str(Path(_REPOSITORY_ROOT) / "cpp_runtime" / "backends" / "hip" / "include")
_CORE_INCLUDE = str(Path(_REPOSITORY_ROOT) / "cpp_runtime" / "core" / "include")
_GGML_ROOT = Path(_REPOSITORY_ROOT) / "cpp_runtime" / "components" / "ggml"
_GGML_INCLUDE = str(_GGML_ROOT / "include")
_GGML_SOURCE_INCLUDE = str(_GGML_ROOT / "src")
_GGML_HIP_INCLUDE = str(_GGML_ROOT / "src" / "ggml-hip")
_GGML_VENDORS_INCLUDE = str(_GGML_ROOT / "src" / "ggml-hip" / "vendors")
_SOURCES = [
    os.path.join(_DIR, "norm.hip"),
    os.path.join(_DIR, "acc.hip"),
    os.path.join(_DIR, "rope.hip"),
    os.path.join(_DIR, "attention.hip"),
    os.path.join(_DIR, "gated_delta_net.hip"),
    os.path.join(_DIR, "kv_cache.hip"),
    os.path.join(_DIR, "activation.hip"),
    os.path.join(_DIR, "embedding.hip"),
    os.path.join(_DIR, "selected_attention.hip"),
    os.path.join(_DIR, "qwen4_exp.hip"),
    os.path.join(_DIR, "glm5_next.hip"),
    os.path.join(_DIR, "sampling.hip"),
    os.path.join(_DIR, "ssm_conv.hip"),
    os.path.join(_DIR, "moe.hip"),
    os.path.join(_DIR, "mx_matmul.hip"),
    os.path.join(_DIR, "fp8_sq.hip"),
    os.path.join(_DIR, "mxfp4_sq.hip"),
    os.path.join(_DIR, "nint_matmul.hip"),
    os.path.join(_DIR, "nvq_matmul.hip"),
    os.path.join(_DIR, "nepq.hip"),
    os.path.join(_DIR, "nepq_residual.hip"),
    os.path.join(_DIR, "mfq_hip.cpp"),
]
_module = None


def ext():
    global _module
    if _module is None:
        import torch
        from torch.utils.cpp_extension import load

        if not torch.version.hip:
            raise RuntimeError("the HIP kernel extension requires a ROCm build of torch")
        _module = load(
            name="mfq_hip",
            sources=_SOURCES,
            extra_include_paths=[
                _DIR,
                _REPOSITORY_ROOT,
                _HIP_RUNTIME_INCLUDE,
                _CORE_INCLUDE,
                _GGML_INCLUDE,
                _GGML_SOURCE_INCLUDE,
                _GGML_HIP_INCLUDE,
                _GGML_VENDORS_INCLUDE,
            ],
            extra_cflags=["-std=c++20"],
            # The CUDA build undefines __CUDA_NO_HALF_OPERATORS__ and friends
            # so the kernels can use half arithmetic directly; mirror that by
            # undoing torch's __HIP_NO_HALF* restrictions. The compat header
            # is force-included first so every translation unit sees the
            # wave64 shuffle mappings and intrinsic shims before any kernel
            # or ggml header.
            extra_cuda_cflags=[
                "-O3",
                "-std=c++20",
                "-U__HIP_NO_HALF_OPERATORS__",
                "-U__HIP_NO_HALF_CONVERSIONS__",
                "-include",
                "mfq_hip_compat.h",
            ],
            extra_ldflags=["-lhipblas"],
            verbose=False,
        )
    return _module
