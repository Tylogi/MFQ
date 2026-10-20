"""Torch reference math for Mamba-2 selective state scans.

This module is deliberately architecture-free: it implements the recurrence
used by any Mamba-2 mixer that stores one scalar decay per head.  Layerwise
calibration uses it instead of the optional CUDA kernels so a Windows CUDA
installation only needs PyTorch.
"""

from __future__ import annotations

import torch


def selective_scan_chunked(
    values: torch.Tensor,
    time_step: torch.Tensor,
    decay: torch.Tensor,
    recurrent_b: torch.Tensor,
    recurrent_c: torch.Tensor,
    *,
    skip: torch.Tensor | None = None,
    chunk_size: int = 128,
    initial_state: torch.Tensor | None = None,
    return_state: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Evaluate ``s_t = exp(dt_t * A) * s_{t-1} + dt_t * B_t (x) x_t``.

    ``values`` is ``(batch, time, heads, head_dim)``, ``time_step`` is
    ``(batch, time, heads)`` already softplus-transformed and clamped,
    ``decay`` is the negative ``(heads,)`` rate, and ``recurrent_b`` /
    ``recurrent_c`` are ``(batch, time, heads, state)``.

    The scan is split into time chunks so the intra-chunk score tensor is
    ``O(chunk_size**2 * heads)`` rather than ``O(time**2 * heads)``.  Each
    chunk performs three tensor contractions over the state axis, so the total
    work is ``O(time * heads * head_dim * state)`` plus the intra-chunk term
    ``O(time * chunk_size * heads * (head_dim + state))``.  All accumulation is
    float32, matching the Transformers torch path.
    """

    if chunk_size <= 0:
        raise ValueError("selective scan chunk size must be positive")
    if values.ndim != 4 or time_step.ndim != 3:
        raise ValueError(
            "selective scan expects values (batch, time, heads, head_dim) and "
            "time_step (batch, time, heads)"
        )
    batch, time, heads, head_dim = values.shape
    state_size = int(recurrent_b.shape[-1])
    if tuple(recurrent_b.shape) != (batch, time, heads, state_size):
        raise ValueError("recurrent_b must match values and the state size")
    if tuple(recurrent_c.shape) != tuple(recurrent_b.shape):
        raise ValueError("recurrent_c must match recurrent_b")
    if tuple(time_step.shape) != (batch, time, heads):
        raise ValueError("time_step must match the value heads")
    if tuple(decay.shape) != (heads,):
        raise ValueError("decay must contain one rate per head")

    rate = decay.to(device=values.device, dtype=torch.float32)
    if initial_state is None:
        state = torch.zeros(
            (batch, heads, head_dim, state_size),
            device=values.device,
            dtype=torch.float32,
        )
    else:
        if tuple(initial_state.shape) != (batch, heads, head_dim, state_size):
            raise ValueError("selective scan initial state does not match the recurrence")
        state = initial_state.to(device=values.device, dtype=torch.float32).clone()
    output = torch.empty(
        (batch, time, heads, head_dim),
        device=values.device,
        dtype=torch.float32,
    )
    triangular = torch.tril(
        torch.ones(
            (chunk_size, chunk_size),
            device=values.device,
            dtype=torch.bool,
        )
    )
    for start in range(0, time, chunk_size):
        end = min(start + chunk_size, time)
        length = end - start
        current = values[:, start:end].to(torch.float32)
        step = time_step[:, start:end].to(torch.float32)
        current_b = recurrent_b[:, start:end].to(torch.float32)
        current_c = recurrent_c[:, start:end].to(torch.float32)

        # ``cumulative`` is non-positive because every rate is negative.
        cumulative = torch.cumsum(step * rate[None, None, :], dim=1)
        scaled_values = current * step.unsqueeze(-1)
        per_head = cumulative.permute(0, 2, 1)
        differences = per_head[:, :, :, None] - per_head[:, :, None, :]
        intra_decay = torch.exp(
            differences.masked_fill(
                ~triangular[:length, :length],
                float("-inf"),
            )
        )
        scores = torch.einsum("bthn,bshn->bhts", current_c, current_b)
        intra = torch.einsum(
            "bhts,bshp->bthp",
            scores * intra_decay,
            scaled_values,
        )
        inter = torch.einsum(
            "bthn,bhpn->bthp",
            current_c * torch.exp(cumulative).unsqueeze(-1),
            state,
        )

        chunk_end = cumulative[:, -1]
        residual_decay = torch.exp(chunk_end[:, None, :] - cumulative)
        weighted_b = current_b * residual_decay.unsqueeze(-1)
        state = torch.exp(chunk_end)[:, :, None, None] * state + torch.einsum(
            "bshn,bshp->bhpn",
            weighted_b,
            scaled_values,
        )
        output[:, start:end] = intra + inter

    if skip is not None:
        if tuple(skip.shape) != (heads,):
            raise ValueError("skip must contain one coefficient per head")
        output = output + (
            values.to(torch.float32) * skip.to(torch.float32)[None, None, :, None]
        )
    if return_state:
        return output, state
    return output


__all__ = ["selective_scan_chunked"]
