"""The unified CUDA extension manifest must name files that exist.

Cheap gate: reading the manifest does not import torch, so this runs without a
CUDA toolchain.  The compile/load/numeric gate is tests/test_activation_bridge_cuda.py.
"""
import os

from mfq.kernels.cuda._ext import _SOURCES


def test_extension_sources_exist():
    missing = [path for path in _SOURCES if not os.path.isfile(path)]
    assert not missing, "extension manifest names missing sources: " + ", ".join(missing)
