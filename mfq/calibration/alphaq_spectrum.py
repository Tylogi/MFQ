"""Batched symmetric CUDA spectra for complete AlphaQ 128-square blocks."""
from __future__ import annotations

import ctypes as C
import ctypes.util
from functools import lru_cache
import os
from pathlib import Path
import sys


@lru_cache(maxsize=1)
def cuda_library():
    """Discover a cuSOLVER exposing the batched generic symmetric API."""
    import torch

    candidates = []
    maps = Path('/proc/self/maps')
    if maps.is_file():
        for line in maps.read_text().splitlines():
            if 'libcusolver.so' in line:
                candidates.append(line.split()[-1])
    found = ctypes.util.find_library('cusolver')
    if found:
        candidates.append(found)
    roots = [Path(torch.__file__).parent/'lib']
    roots.extend(Path(p)/'nvidia/cusolver/lib' for p in sys.path if p)
    for variable in ('CUDA_PATH', 'CUDA_HOME'):
        if os.environ.get(variable):
            roots.extend(Path(os.environ[variable])/p for p in ('lib64', 'bin'))
    roots.extend(p/'lib64' for p in sorted(Path('/usr/local').glob('cuda*'), reverse=True))
    for root in roots:
        for pattern in ('libcusolver.so*', 'cusolver64_*.dll'):
            candidates.extend(str(p) for p in sorted(root.glob(pattern), reverse=True))
    pointer, integer, wide, size = C.c_void_p, C.c_int, C.c_int64, C.c_size_t
    declarations = {
        'cusolverDnCreate': [C.POINTER(pointer)],
        'cusolverDnDestroy': [pointer],
        'cusolverDnSetStream': [pointer, pointer],
        'cusolverDnXsyevBatched_bufferSize': [pointer, pointer, integer, integer, wide,
            integer, pointer, wide, integer, pointer, integer, C.POINTER(size), C.POINTER(size), wide],
        'cusolverDnXsyevBatched': [pointer, pointer, integer, integer, wide, integer,
            pointer, wide, integer, pointer, integer, pointer, size, pointer, size, pointer, wide],
    }
    for candidate in dict.fromkeys(candidates):
        try:
            library = C.CDLL(candidate)
            functions = {name: getattr(library, name) for name in declarations}
        except (OSError, AttributeError):
            continue
        for name, function in functions.items():
            function.argtypes = declarations[name]
            function.restype = integer
        return library
    return None


def _check(status):
    if status:
        raise RuntimeError(f'cuSOLVER batched spectrum status {status}')


def symmetric_eigenvalues(gram):
    """Return FP32 eigenvalues, preserving the input and current CUDA stream."""
    import torch

    if gram.ndim != 3 or gram.dtype != torch.float32 or gram.shape[-1] != gram.shape[-2]:
        raise ValueError('FP32 batches of square symmetric matrices required')
    if not gram.is_cuda:
        return torch.linalg.eigvalsh(gram)
    library = cuda_library()
    if library is None:
        raise RuntimeError('installed cuSOLVER has no generic batched symmetric eigensolver')
    batch, n, _ = gram.shape
    if batch < 1 or n < 1 or batch*n*n > 2147483647:
        raise ValueError('cuSOLVER batched matrix geometry exceeds its integer indexing')
    with torch.cuda.device(gram.device):
        # Symmetry makes row-major and column-major representations identical.
        matrix = gram.clone().contiguous()
        values = torch.empty((batch, n), dtype=gram.dtype, device=gram.device)
        info = torch.empty(batch, dtype=torch.int32, device=gram.device)
        handle = C.c_void_p()
        _check(library.cusolverDnCreate(C.byref(handle)))
        try:
            _check(library.cusolverDnSetStream(handle, C.c_void_p(torch.cuda.current_stream().cuda_stream)))
            args = (handle, None, 0, 0, n, 0, matrix.data_ptr(), n, 0, values.data_ptr(), 0)
            device_bytes, host_bytes = C.c_size_t(), C.c_size_t()
            _check(library.cusolverDnXsyevBatched_bufferSize(
                *args, C.byref(device_bytes), C.byref(host_bytes), batch))
            workspace = torch.empty(device_bytes.value, dtype=torch.uint8, device=gram.device)
            host = C.create_string_buffer(max(1, host_bytes.value))
            _check(library.cusolverDnXsyevBatched(*args, workspace.data_ptr(), device_bytes.value,
                host, host_bytes.value, info.data_ptr(), batch))
            if bool((info != 0).any()):
                raise ArithmeticError('batched symmetric eigensolver did not converge')
            return values
        finally:
            _check(library.cusolverDnDestroy(handle))


def gram_eigenvalues(blocks):
    """Use full FP32 products; never inherit a caller's TF32 setting."""
    import torch

    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        gram = torch.bmm(blocks, blocks.transpose(-1, -2))
        return symmetric_eigenvalues(gram).clamp_min_(0)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
