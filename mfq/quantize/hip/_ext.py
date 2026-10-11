"""Loader for the offline quantization HIP extension on AMD ROCm hosts.

Mirrors mfq.quantize.cuda._ext but compiles the hipify-translated kernels in
this directory. Routed to automatically by mfq.quantize.cuda._ext.ext() when
torch reports a HIP build.
"""

from __future__ import annotations

from pathlib import Path


_DIR = Path(__file__).resolve().parent
_SOURCES = [
    str(_DIR / "nvq_quant_assign.hip"),
    str(_DIR / "nepq0_s_assign.hip"),
    str(_DIR / "npq0_s_assign.hip"),
    str(_DIR / "npq0_l_assign.hip"),
    str(_DIR / "nvq2j_assign.hip"),
    str(_DIR / "nvq3j_assign.hip"),
    str(_DIR / "nvq_pack.hip"),
    str(_DIR / "nint_quant.hip"),
    str(_DIR / "nvq_quant_hip.cpp"),
]
_module = None


def ext():
    global _module
    if _module is None:
        import torch
        from torch.utils.cpp_extension import load

        if not torch.version.hip:
            raise RuntimeError(
                "the HIP quantization extension requires a ROCm build of torch"
            )
        _module = load(
            name="mfq_nvq_quant_hip",
            sources=_SOURCES,
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    return _module
