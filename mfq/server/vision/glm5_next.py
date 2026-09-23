"""GLM-5.3-Flash vision processor."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from mfq.server.vision.decode import VisionProcessingError
from mfq.server.vision.flash_next import _FlashNextImageProcessor
from mfq.server.vision.video import _DecodedVideo


class Glm5NextVisionProcessor(_FlashNextImageProcessor):
    """GLM-5.3-Flash media path matching Transformers 5.16.1."""

    processor_name = "glm5_next"
    patch_size = 14
    image_mean = (0.48145466, 0.4578275, 0.40821073)
    image_std = (0.26862954, 0.26130258, 0.27577711)
    minimum_tokens = 16
    maximum_tokens = 8_000
    maximum_video_tokens = 240_000
    maximum_video_frames = 2_048
    forbidden_text_tokens = (
        "<|begin_of_image|>",
        "<|image|>",
        "<|end_of_image|>",
        "<|begin_of_video|>",
        "<|video|>",
        "<|end_of_video|>",
    )

    @classmethod
    def _smart_resize(
        cls,
        frames: int,
        height: int,
        width: int,
        *,
        maximum_tokens: int | None = None,
    ) -> tuple[int, int]:
        temporal = cls.temporal_patch_size
        factor = cls.patch_size * cls.merge_size
        pixels_per_token = temporal * factor * factor
        minimum_pixels = cls.minimum_tokens * pixels_per_token
        maximum_pixels = (
            cls.maximum_tokens if maximum_tokens is None else maximum_tokens
        ) * pixels_per_token

        def align(value: int) -> int:
            return math.ceil(value / factor) * factor

        aligned_frames = max(temporal, round(frames / temporal) * temporal)
        aligned_height = align(height)
        aligned_width = align(width)
        budget = aligned_frames * aligned_height * aligned_width
        if budget < minimum_pixels:
            scale = math.sqrt(minimum_pixels / (frames * height * width))
            aligned_height = align(max(1, math.ceil(height * scale)))
            aligned_width = align(max(1, math.ceil(width * scale)))
            budget = aligned_frames * aligned_height * aligned_width
        if budget > maximum_pixels:
            if maximum_pixels < aligned_frames * factor * factor:
                raise VisionProcessingError("GLM-5-Next image token budget is too small")
            low, high = 1, height
            best_height = best_width = factor
            while low <= high:
                content_height = (low + high) // 2
                content_width = max(1, math.floor(width * content_height / height))
                candidate_height = align(content_height)
                candidate_width = align(content_width)
                if aligned_frames * candidate_height * candidate_width <= maximum_pixels:
                    best_height, best_width = candidate_height, candidate_width
                    low = content_height + 1
                else:
                    high = content_height - 1
            aligned_height, aligned_width = best_height, best_width
        return aligned_height, aligned_width

    @classmethod
    def _prepare_image(cls, image: Any) -> tuple[np.ndarray, tuple[int, int, int]]:
        from PIL import Image

        image = image.convert("RGB")
        height, width = image.height, image.width
        if height <= 0 or width <= 0:
            raise VisionProcessingError("GLM-5-Next image dimensions must be positive")
        target_height, target_width = cls._smart_resize(
            cls.temporal_patch_size,
            height,
            width,
        )
        factor = cls.patch_size * cls.merge_size
        pixels_per_token = cls.temporal_patch_size * factor * factor
        scale = min(target_height / height, target_width / width)
        if cls.temporal_patch_size * height * width >= pixels_per_token * cls.minimum_tokens:
            scale = min(1.0, scale)
        content_height = max(1, min(target_height, math.floor(height * scale)))
        content_width = max(1, min(target_width, math.floor(width * scale)))
        if image.size != (content_width, content_height):
            image = image.resize(
                (content_width, content_height),
                resample=Image.Resampling.BICUBIC,
            )
        pixels = np.asarray(image, dtype=np.uint8).transpose(2, 0, 1)
        pixels = np.pad(
            pixels,
            (
                (0, 0),
                (0, target_height - content_height),
                (0, target_width - content_width),
            ),
            mode="constant",
        )
        return cls._patchify(cls._normalize(pixels))

    @classmethod
    def _placeholder(cls, merged_tokens: int) -> str:
        return "<|begin_of_image|>" + "<|image|>" * merged_tokens + "<|end_of_image|>"

    @classmethod
    def _sample_video_indices(
        cls,
        total_frames: int,
        frames_per_second: float,
        duration_seconds: float,
    ) -> tuple[int, ...]:
        if total_frames <= 0:
            raise VisionProcessingError("GLM-5-Next video contains no frames")
        maximum_index = total_frames - 1
        duration = (
            duration_seconds
            if duration_seconds > 0.0
            else round(maximum_index / frames_per_second) + 1
        )
        maximum_seconds = int(duration)
        requested = max(1, int(duration * cls.video_fps))
        requested = min(requested, cls.maximum_video_frames)
        timestamps = [index / frames_per_second for index in range(total_frames)]
        if total_frames < requested:
            indices = np.linspace(0, total_frames - 1, requested, dtype=int).tolist()
        else:
            indices = []
            current_second = 0.0
            increment = 1.0 / cls.video_fps
            for index, timestamp in enumerate(timestamps):
                if timestamp >= current_second:
                    current_second += increment
                    indices.append(index)
                    if current_second >= maximum_seconds:
                        break
        if len(indices) < requested:
            start = indices[0] if indices else 0
            end = indices[-1] if indices else max(total_frames - 1, 0)
            indices = np.linspace(start, end, requested, dtype=int).tolist()
        elif len(indices) > requested:
            indices = np.linspace(0, total_frames - 1, requested, dtype=int).tolist()
        unique: list[int] = []
        seen: set[int] = set()
        for value in indices:
            if value not in seen:
                seen.add(value)
                unique.append(int(value))
        if len(unique) % cls.temporal_patch_size:
            unique.extend(unique[-1] for _ in range(cls.temporal_patch_size - len(unique) % cls.temporal_patch_size))
        return tuple(unique)

    @classmethod
    def _prepare_video(
        cls,
        decoded: _DecodedVideo,
    ) -> tuple[np.ndarray, tuple[int, int, int], str]:
        from PIL import Image

        if not decoded.frames:
            raise VisionProcessingError("GLM-5-Next video contains no decoded frames")
        width, height = decoded.frames[0].source_size
        if any(frame.source_size != (width, height) for frame in decoded.frames):
            raise VisionProcessingError("GLM-5-Next video changes dimensions between frames")
        frame_count = len(decoded.frames)
        target_height, target_width = cls._smart_resize(
            frame_count,
            height,
            width,
            maximum_tokens=cls.maximum_video_tokens,
        )
        factor = cls.patch_size * cls.merge_size
        pixels_per_token = cls.temporal_patch_size * factor * factor
        scale = min(target_height / height, target_width / width)
        if frame_count * height * width >= pixels_per_token * cls.minimum_tokens:
            scale = min(1.0, scale)
        content_height = max(1, min(target_height, math.floor(height * scale)))
        content_width = max(1, min(target_width, math.floor(width * scale)))
        frames: list[np.ndarray] = []
        for prepared in decoded.frames:
            image = prepared.image.convert("RGB")
            if image.size != (content_width, content_height):
                image = image.resize(
                    (content_width, content_height),
                    resample=Image.Resampling.BICUBIC,
                )
            pixels = np.asarray(image, dtype=np.uint8).transpose(2, 0, 1)
            pixels = np.pad(
                pixels,
                (
                    (0, 0),
                    (0, target_height - content_height),
                    (0, target_width - content_width),
                ),
                mode="constant",
            )
            frames.append(cls._normalize(pixels))
        pixel_values, grid = cls._patchify_video(np.stack(frames, axis=0))
        frame_tokens = grid[1] * grid[2] // (cls.merge_size**2)
        timestamps = [
            value / decoded.frames_per_second
            for value in decoded.frame_indices[:: cls.temporal_patch_size]
        ]
        timestamps = timestamps[: grid[0]]
        while len(timestamps) < grid[0]:
            timestamps.append(timestamps[-1] if timestamps else 0.0)
        placeholder = "<|begin_of_video|>" + "".join(
            "<|begin_of_image|>"
            + "<|image|>" * frame_tokens
            + "<|end_of_image|>"
            + f"{timestamp:.1f} seconds"
            for timestamp in timestamps
        ) + "<|end_of_video|>"
        return pixel_values, grid, placeholder
