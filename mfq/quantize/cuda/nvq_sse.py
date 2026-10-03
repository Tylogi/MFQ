"""Fused fixed-table NVQ reconstruction error without CPU code streams."""
import torch
import triton
import triton.language as tl


@triton.jit
def _jsc_sse(X, W, Anchor, State, Indices, Alpha, Bank, Codebook, Out,
             WIDTH: tl.constexpr, NG: tl.constexpr, VS: tl.constexpr,
             ENTRIES: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    valid = col < WIDTH
    state = tl.load(State + row * NG + col // 24, valid, 0).to(tl.int32)
    bank = tl.load(Bank + state).to(tl.int32)
    index = tl.load(Indices + row * (WIDTH // VS) + col // VS, valid, 0).to(tl.int32)
    code = tl.load(Codebook + (bank * ENTRIES + index) * VS + col % VS, valid, 0).to(tl.float32)
    scale = tl.load(Anchor + row) * tl.load(Alpha + state)
    reconstruction = code * scale
    error = tl.load(X + row * WIDTH + col, valid, 0) - reconstruction
    weight = tl.load(W + row * WIDTH + col, valid, 0)
    tl.store(Out + row, tl.sum(weight * error * error, 0))


def jsc_row_sse(target, weight, anchor, state, indices, alpha, bank, codebooks, vector_size):
    rows, width = target.shape
    result = torch.empty(rows, device=target.device, dtype=torch.float32)
    _jsc_sse[(rows,)](
        target, weight, anchor, state, indices, alpha, bank, codebooks, result,
        width, width // 24, vector_size, codebooks.shape[1], triton.next_power_of_2(width),
        enable_fp_fusion=False,
    )
    return result
