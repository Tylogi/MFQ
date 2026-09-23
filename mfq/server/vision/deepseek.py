"""DeepSeek V4 / V4.1 image processors."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import numpy as np

from mfq.server.vision.decode import (
    ProcessedVisionRequest,
    VisionProcessingError,
    _binary_tensors,
    _decode_data_url,
    _decode_image,
    _tensor,
)


class DeepseekV4VisionProcessor:
    """Exact image preprocessing contract released with DSV4 Vision.

    The processor intentionally leaves ``<｜deepseek_image｜>`` in the prompt.
    The final image block depends on the marker's *token* position, so the
    native worker expands it only after applying the chat template and
    tokenizing the prompt.
    """

    image_placeholder = "<｜deepseek_image｜>"
    processor_name = "deepseek_v4"
    model_label = "DeepSeek-V4"
    patch_size = 14
    downsample_ratio = 3
    maximum_image_tokens = 384
    minimum_pixels = 147_456
    maximum_width_height_ratio = 8

    @classmethod
    def _grid_tokens(
        cls,
        best_height: int,
        best_width: int,
    ) -> tuple[int, int, int]:
        n_llm_h = math.ceil((best_height // cls.patch_size) / cls.downsample_ratio)
        n_llm_w = math.ceil((best_width // cls.patch_size) / cls.downsample_ratio)
        count = n_llm_h * (n_llm_w + 1) + 2
        if n_llm_h % 2 == 1:
            count += n_llm_w + 1
        count += (n_llm_h + 1) // 2 * (n_llm_w + 1) % 2 * 2
        return n_llm_h, n_llm_w, count

    @classmethod
    def _solve_resize_ratio(
        cls,
        height: int,
        width: int,
        budget: int,
    ) -> tuple[int, int, int, int, int]:
        ratio = height / width
        max_w_float = math.sqrt((budget - 2) / ratio + 0.25) - 0.5
        max_h_float = max_w_float * ratio
        unit = cls.patch_size * cls.downsample_ratio
        if max_w_float < 1.0:
            max_w = 1
            max_h = (budget - 2) // (max_w + 1)
            if max_h % 2 == 1:
                max_h -= 1
            best_width = max_w * unit
            best_height = max_h * unit
        elif max_h_float < 2.0:
            max_h = 2
            max_w = (budget - 2) // max_h - 1
            if max_w <= 1:
                raise VisionProcessingError("image token budget is too small")
            best_width = max_w * unit
            best_height = max_h * unit
        else:
            max_w = math.floor(max_w_float)
            max_h = math.floor(max_h_float)
            if max_h % 2 == 1:
                max_h -= 1
            beta = min(max_w * unit / width, max_h * unit / height)
            best_width = math.floor(width * beta / cls.patch_size) * cls.patch_size
            best_height = math.floor(height * beta / cls.patch_size) * cls.patch_size
        n_llm_h, n_llm_w, count = cls._grid_tokens(best_height, best_width)
        return n_llm_h, n_llm_w, best_height, best_width, count

    @classmethod
    def _safe_resize(
        cls,
        height: int,
        width: int,
        best_height: int,
        best_width: int,
    ) -> tuple[int, int, int, int]:
        maximum = cls.maximum_image_tokens - 3
        n_llm_h, n_llm_w, count = cls._grid_tokens(best_height, best_width)
        budget = maximum
        while count > maximum:
            n_llm_h, n_llm_w, best_height, best_width, count = cls._solve_resize_ratio(
                height, width, budget
            )
            budget -= 1
            if budget <= 4:
                raise VisionProcessingError("unable to fit image in the token budget")
        return n_llm_h, n_llm_w, best_height, best_width

    @classmethod
    def _prepare_image(
        cls,
        image: Any,
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        from PIL import ImageOps

        image = image.convert("RGB")
        source_width, source_height = image.size
        if source_width <= 0 or source_height <= 0:
            raise VisionProcessingError("image dimensions must be positive")
        width, height = source_width, source_height
        if (cls.maximum_width_height_ratio is not None and
                width > height * cls.maximum_width_height_ratio):
            width = height * cls.maximum_width_height_ratio
        if width * height < cls.minimum_pixels:
            ratio = math.sqrt(cls.minimum_pixels / (width * height))
            width = int(width * ratio)
            height = int(height * ratio)
        best_width = math.ceil(width / cls.patch_size) * cls.patch_size
        best_height = math.ceil(height / cls.patch_size) * cls.patch_size
        n_llm_h, n_llm_w, best_height, best_width = cls._safe_resize(
            height,
            width,
            best_height,
            best_width,
        )
        n_vit_h = best_height // cls.patch_size
        n_vit_w = best_width // cls.patch_size
        if (cls.maximum_width_height_ratio is not None and
                source_width >= cls.maximum_width_height_ratio * source_height):
            image = image.resize((best_width, best_height))
        else:
            image = ImageOps.pad(
                image,
                (best_width, best_height),
                color=(127, 127, 127),
            )
        pixels = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)
        pixels = (pixels / np.float32(255.0) - np.float32(0.5)) / np.float32(0.5)
        patches = (
            pixels.reshape(
                3,
                n_vit_h,
                cls.patch_size,
                n_vit_w,
                cls.patch_size,
            )
            .transpose(1, 3, 0, 2, 4)
            .reshape(-1, 3, cls.patch_size, cls.patch_size)
        )
        return (
            np.ascontiguousarray(patches, dtype=np.float32),
            (n_vit_h, n_vit_w, n_llm_h, n_llm_w),
        )

    @classmethod
    def _pack_tensors(
        cls,
        patches: list[np.ndarray],
        grids: list[tuple[int, int, int, int]],
        *,
        use_binary_file: bool,
    ) -> tuple[dict[str, Any], tuple[Path, ...]]:
        if not patches or len(patches) != len(grids):
            raise VisionProcessingError("processed image tensor geometry is invalid")
        maximum_patches = max(value.shape[0] for value in patches)
        pixels = np.zeros(
            (
                len(patches),
                maximum_patches,
                3 * cls.patch_size * cls.patch_size,
            ),
            dtype=np.float32,
        )
        mask = np.zeros((len(patches), maximum_patches), dtype=np.uint8)
        for index, value in enumerate(patches):
            pixels[index, : value.shape[0]] = value.reshape(value.shape[0], -1)
            mask[index, : value.shape[0]] = 1
        values = [
            ("pixel_values", pixels, "float32"),
            ("patch_mask", mask, "uint8"),
            ("vision_grid", np.asarray(grids, dtype=np.int32), "int32"),
        ]
        if use_binary_file:
            tensors, path = _binary_tensors(values)
            tensors["version"] = 2
            tensors["processor"] = cls.processor_name
            return tensors, (path,)
        return (
            {
                "version": 2,
                "processor": cls.processor_name,
                **{
                    name: _tensor(value, dtype)
                    for name, value, dtype in values
                },
            },
            (),
        )

    def prepare_openai_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        use_binary_file: bool = False,
    ) -> ProcessedVisionRequest | None:
        prepared = copy.deepcopy(messages)
        patches: list[np.ndarray] = []
        grids: list[tuple[int, int, int, int]] = []
        sources = 0
        found_media = False
        for message in prepared:
            content = message.get("content")
            if not isinstance(content, list):
                if isinstance(content, str) and self.image_placeholder in content:
                    raise VisionProcessingError(
                        "image placeholder tokens cannot be supplied as text"
                    )
                continue
            pieces: list[str] = []
            for item in content:
                if not isinstance(item, dict):
                    raise VisionProcessingError("multimodal content item must be an object")
                item_type = item.get("type")
                if item_type == "text":
                    value = str(item.get("text", ""))
                    if self.image_placeholder in value:
                        raise VisionProcessingError(
                            "image placeholder tokens cannot be supplied as text"
                        )
                    pieces.append(value)
                    continue
                if item_type != "image_url":
                    raise VisionProcessingError(
                        f"{self.model_label} Vision does not support "
                        f"{item_type or 'unknown'} input"
                    )
                image_spec = item.get("image_url")
                if not isinstance(image_spec, dict) or not isinstance(image_spec.get("url"), str):
                    raise VisionProcessingError("image_url content is missing its URL")
                image = _decode_image(
                    _decode_data_url(
                        image_spec["url"],
                        "image/",
                    )
                )
                value, grid = self._prepare_image(image)
                patches.append(value)
                grids.append(grid)
                pieces.append(self.image_placeholder)
                sources += 1
                found_media = True
            message["content"] = "\n\n".join(piece for piece in pieces if piece)
        if not found_media:
            return None
        tensors, cleanup_paths = self._pack_tensors(
            patches,
            grids,
            use_binary_file=use_binary_file,
        )
        return ProcessedVisionRequest(
            messages=prepared,
            tensors=tensors,
            source_count=sources,
            frame_count=0,
            cleanup_paths=cleanup_paths,
        )


class DeepseekV41VisionProcessor(DeepseekV4VisionProcessor):
    """Released DeepSeek-V4.1 patch and row-major image-span contract."""

    processor_name = "deepseek_v41"
    model_label = "DeepSeek-V4.1"
    maximum_image_tokens = 1024
    minimum_pixels = 544 * 544
    maximum_width_height_ratio = None

    @classmethod
    def _grid_tokens(
        cls,
        best_height: int,
        best_width: int,
    ) -> tuple[int, int, int]:
        n_llm_h = math.ceil(
            (best_height // cls.patch_size) / cls.downsample_ratio
        )
        n_llm_w = math.ceil(
            (best_width // cls.patch_size) / cls.downsample_ratio
        )
        return n_llm_h, n_llm_w, n_llm_h * (n_llm_w + 1) + 2

    @classmethod
    def _solve_resize_ratio(
        cls,
        height: int,
        width: int,
        budget: int,
    ) -> tuple[int, int, int, int, int]:
        ratio = height / width
        max_w_float = math.sqrt((budget - 2) / ratio + 0.25) - 0.5
        max_h_float = max_w_float * ratio
        unit = cls.patch_size * cls.downsample_ratio
        if max_w_float < 1.0:
            best_height = (budget - 2) // 2 * unit
            best_width = unit
        elif max_h_float < 1.0:
            best_height = unit
            best_width = (budget - 3) * unit
        else:
            beta = min(
                math.floor(max_w_float) * unit / width,
                math.floor(max_h_float) * unit / height,
            )
            best_height = math.floor(height * beta / cls.patch_size) * cls.patch_size
            best_width = math.floor(width * beta / cls.patch_size) * cls.patch_size
        n_llm_h, n_llm_w, count = cls._grid_tokens(best_height, best_width)
        return n_llm_h, n_llm_w, best_height, best_width, count

    @classmethod
    def _safe_resize(
        cls,
        height: int,
        width: int,
        best_height: int,
        best_width: int,
    ) -> tuple[int, int, int, int]:
        n_llm_h, n_llm_w, count = cls._grid_tokens(best_height, best_width)
        if count > cls.maximum_image_tokens:
            n_llm_h, n_llm_w, best_height, best_width, count = (
                cls._solve_resize_ratio(
                    height, width, cls.maximum_image_tokens
                )
            )
        if count > cls.maximum_image_tokens:
            raise VisionProcessingError("unable to fit image in the token budget")
        return n_llm_h, n_llm_w, best_height, best_width
