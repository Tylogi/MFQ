"""Qwen4-Exp / Qwen3.5 vision processors (shared Qwen tower contract)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from mfq.server.vision.decode import VisionProcessingError
from mfq.server.vision.flash_next import _FlashNextImageProcessor
from mfq.server.vision.video import _DecodedVideo


class Qwen4ExpVisionProcessor(_FlashNextImageProcessor):
    """Qwen3.8-Flash-Next media path matching its Qwen3-VL processor."""

    processor_name = "qwen4_exp"
    patch_size = 16
    image_mean = (0.5, 0.5, 0.5)
    image_std = (0.5, 0.5, 0.5)
    minimum_pixels = 65_536
    maximum_pixels = 16_777_216
    minimum_video_pixels = 4_096
    maximum_video_pixels = 25_165_824
    minimum_video_frames = 4
    maximum_video_frames = 768
    forbidden_text_tokens = (
        "<|vision_start|>",
        "<|image_pad|>",
        "<|video_pad|>",
        "<|vision_end|>",
    )

    @classmethod
    def _smart_resize(cls, height: int, width: int) -> tuple[int, int]:
        factor = cls.patch_size * cls.merge_size
        if min(height, width) <= 0 or max(height, width) / min(height, width) > 200:
            raise VisionProcessingError("Qwen4-Exp image aspect ratio is invalid")
        resized_height = round(height / factor) * factor
        resized_width = round(width / factor) * factor
        if resized_height * resized_width > cls.maximum_pixels:
            beta = math.sqrt((height * width) / cls.maximum_pixels)
            resized_height = max(factor, math.floor(height / beta / factor) * factor)
            resized_width = max(factor, math.floor(width / beta / factor) * factor)
        elif resized_height * resized_width < cls.minimum_pixels:
            beta = math.sqrt(cls.minimum_pixels / (height * width))
            resized_height = math.ceil(height * beta / factor) * factor
            resized_width = math.ceil(width * beta / factor) * factor
        return resized_height, resized_width

    @classmethod
    def _prepare_image(cls, image: Any) -> tuple[np.ndarray, tuple[int, int, int]]:
        from PIL import Image

        image = image.convert("RGB")
        resized_height, resized_width = cls._smart_resize(image.height, image.width)
        if image.size != (resized_width, resized_height):
            image = image.resize(
                (resized_width, resized_height),
                resample=Image.Resampling.BICUBIC,
            )
        pixels = np.asarray(image, dtype=np.uint8).transpose(2, 0, 1)
        return cls._patchify(cls._normalize(pixels))

    @classmethod
    def _placeholder(cls, merged_tokens: int) -> str:
        return "<|vision_start|>" + "<|image_pad|>" * merged_tokens + "<|vision_end|>"

    @classmethod
    def _sample_video_indices(
        cls,
        total_frames: int,
        frames_per_second: float,
        _duration_seconds: float,
    ) -> tuple[int, ...]:
        if total_frames <= 0:
            raise VisionProcessingError("Qwen4-Exp video contains no frames")
        requested = int(total_frames / frames_per_second * cls.video_fps)
        requested = min(
            max(requested, cls.minimum_video_frames),
            cls.maximum_video_frames,
            total_frames,
        )
        indices = np.linspace(0, total_frames - 1, requested).round().astype(np.int64)
        return tuple(int(value) for value in indices)

    @classmethod
    def _smart_resize_video(
        cls,
        frames: int,
        height: int,
        width: int,
    ) -> tuple[int, int]:
        factor = cls.patch_size * cls.merge_size
        if height < factor or width < factor:
            raise VisionProcessingError(
                "Qwen4-Exp video height and width must be at least one merged patch"
            )
        if max(height, width) / min(height, width) > 200:
            raise VisionProcessingError("Qwen4-Exp video aspect ratio is invalid")
        resized_height = round(height / factor) * factor
        resized_width = round(width / factor) * factor
        aligned_frames = math.ceil(frames / cls.temporal_patch_size) * cls.temporal_patch_size
        budget = aligned_frames * resized_height * resized_width
        if budget > cls.maximum_video_pixels:
            beta = math.sqrt((frames * height * width) / cls.maximum_video_pixels)
            resized_height = max(factor, math.floor(height / beta / factor) * factor)
            resized_width = max(factor, math.floor(width / beta / factor) * factor)
        elif budget < cls.minimum_video_pixels:
            beta = math.sqrt(cls.minimum_video_pixels / (frames * height * width))
            resized_height = math.ceil(height * beta / factor) * factor
            resized_width = math.ceil(width * beta / factor) * factor
        return resized_height, resized_width

    @classmethod
    def _prepare_video(
        cls,
        decoded: _DecodedVideo,
    ) -> tuple[np.ndarray, tuple[int, int, int], str]:
        from PIL import Image

        if not decoded.frames:
            raise VisionProcessingError("Qwen4-Exp video contains no decoded frames")
        width, height = decoded.frames[0].source_size
        if any(frame.source_size != (width, height) for frame in decoded.frames):
            raise VisionProcessingError("Qwen4-Exp video changes dimensions between frames")
        resized_height, resized_width = cls._smart_resize_video(
            len(decoded.frames),
            height,
            width,
        )
        frames: list[np.ndarray] = []
        for prepared in decoded.frames:
            image = prepared.image.convert("RGB")
            if image.size != (resized_width, resized_height):
                image = image.resize(
                    (resized_width, resized_height),
                    resample=Image.Resampling.BICUBIC,
                )
            pixels = np.asarray(image, dtype=np.uint8).transpose(2, 0, 1)
            frames.append(cls._normalize(pixels))
        pixel_values, grid = cls._patchify_video(np.stack(frames, axis=0))
        frame_tokens = grid[1] * grid[2] // (cls.merge_size**2)
        indices = list(decoded.frame_indices)
        if padding := -len(indices) % cls.temporal_patch_size:
            indices.extend(indices[-1] for _ in range(padding))
        timestamps = [value / decoded.frames_per_second for value in indices]
        timestamps = [
            (
                timestamps[index]
                + timestamps[index + cls.temporal_patch_size - 1]
            )
            / cls.temporal_patch_size
            for index in range(0, len(timestamps), cls.temporal_patch_size)
        ]
        placeholder = "".join(
            f"<{timestamp:.1f} seconds>"
            + "<|vision_start|>"
            + "<|video_pad|>" * frame_tokens
            + "<|vision_end|>"
            for timestamp in timestamps
        )
        if len(timestamps) != grid[0]:
            raise VisionProcessingError("Qwen4-Exp video timestamps disagree with temporal grid")
        return pixel_values, grid, placeholder


class Qwen35VisionProcessor(Qwen4ExpVisionProcessor):
    """Qwen3.5-VL media contract; tensor geometry matches Qwen's shared tower."""

    processor_name = "qwen3_5"
