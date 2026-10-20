"""Separable antialiased bicubic resampling without a torch dependency.

The reference is ``torch.nn.functional.interpolate(pixels, size=(h, w),
mode="bicubic", align_corners=False, antialias=True)``, which some checkpoints'
image processors require because plain PIL bicubic changes the pixel values
enough to move the vision tower's output.  ATen's implementation (see
``aten/src/ATen/native/cuda/UpSample.cuh`` and ``HelperInterpCubic``) is a
separable convolution with:

* ``scale = input_size / output_size`` (``area_pixel_compute_scale`` for
  ``align_corners=False`` and no explicit ``scale_factor``);
* source coordinate ``(o + 0.5) * scale - 0.5`` per output index, clamped at
  zero, which is what makes the mapping index-aligned for upsampling;
* the cubic convolution kernel with ``a = -0.5``;
* a support of ``2 * scale`` when ``scale > 1`` (the kernel is evaluated at
  ``(i - center) / scale``, i.e. widened to the source footprint) and a support
  of ``2`` otherwise, so upsampling degenerates to plain bicubic;
* taps restricted to the valid input range and the remaining weights divided by
  their sum, which reproduces ATen's border rows and columns.

ATen evaluates the scale and the source coordinate in the accumulator type of
the input (``float`` for ``float32`` images), so the geometry below is computed
in ``float32`` as well; doing it in ``float64`` shifts the weights by ~5e-5 for
long axes and then shows up as a visible pixel difference.

The two axes are resampled independently with ``float32`` accumulation.
"""

from __future__ import annotations

import numpy as np

_CUBIC_A = -0.5


def _bicubic_kernel(distance: np.ndarray) -> np.ndarray:
    """Evaluate the cubic convolution kernel (``a = -0.5``) at ``distance``."""

    x = np.abs(np.asarray(distance, dtype=np.float32))
    near = x < np.float32(1.0)
    far = (x >= np.float32(1.0)) & (x < np.float32(2.0))
    return np.where(
        near,
        ((_CUBIC_A + 2.0) * x - (_CUBIC_A + 3.0)) * x * x + 1.0,
        np.where(far, (((x - 5.0) * x + 8.0) * x - 4.0) * _CUBIC_A, 0.0),
    ).astype(np.float32)


def _axis_taps(input_size: int, output_size: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return ``(indices, weights)`` for every output position of one axis."""

    if input_size <= 0 or output_size <= 0:
        raise ValueError("bicubic resize needs positive extents")
    scale = np.float32(input_size) / np.float32(output_size)
    support = np.float32(2.0) * scale if scale > np.float32(1.0) else np.float32(2.0)
    kernel_scale = max(scale, np.float32(1.0))
    half = np.float32(0.5)
    taps: list[tuple[np.ndarray, np.ndarray]] = []
    for output_index in range(output_size):
        center = (np.float32(output_index) + half) * scale - half
        first = max(int(np.ceil(center - support)), 0)
        last = min(int(np.floor(center + support)), input_size - 1)
        indices = np.arange(first, last + 1, dtype=np.int32)
        weights = _bicubic_kernel((indices.astype(np.float32) - center) / kernel_scale)
        taps.append((indices, weights / weights.sum(dtype=np.float32)))
    return taps


def antialiased_bicubic_resize(
    pixels: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    """Resize a ``[channels, height, width]`` float32 image, antialiasing on shrink."""

    source = np.asarray(pixels, dtype=np.float32)
    if source.ndim != 3:
        raise ValueError("bicubic resize expects a [channels, height, width] tensor")
    channels, source_height, source_width = source.shape
    if (source_height, source_width) == (height, width):
        return np.ascontiguousarray(source, dtype=np.float32)
    rows = _axis_taps(source_height, height)
    columns = _axis_taps(source_width, width)
    vertical = np.empty((channels, height, source_width), dtype=np.float32)
    for output_index, (indices, weights) in enumerate(rows):
        vertical[:, output_index, :] = np.tensordot(
            weights,
            source[:, indices, :],
            axes=([0], [1]),
        )
    resized = np.empty((channels, height, width), dtype=np.float32)
    for output_index, (indices, weights) in enumerate(columns):
        resized[:, :, output_index] = np.tensordot(
            weights,
            vertical[:, :, indices],
            axes=([0], [2]),
        )
    return resized


__all__ = ["antialiased_bicubic_resize"]
