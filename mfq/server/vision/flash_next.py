"""Shared Flash-Next family image/video preprocessing base."""

from __future__ import annotations

import copy
import io
import math
import os
import platform
from pathlib import Path
from typing import Any

import numpy as np

from mfq.architectures.tensor_schema import GRID_VISION_INPUT_CONTRACT
from mfq.server.vision.decode import (
    ProcessedVisionRequest,
    VisionProcessingError,
    _binary_tensors,
    _decode_data_url,
    _decode_image,
    _tensor,
)
from mfq.server.vision.video import (
    _AVFoundationVideoDecoder,
    _DecodedVideo,
    _PreparedVideoFrame,
)


class _FlashNextImageProcessor:
    """Dependency-light image/video preprocessing for the new VLM families."""

    processor_name = ""
    # Normalized patches and THW grids share one wire contract even though
    # image normalization and prompt markers remain family adapters.
    input_contract = GRID_VISION_INPUT_CONTRACT
    patch_size = 0
    temporal_patch_size = 2
    merge_size = 2
    image_mean: tuple[float, float, float]
    image_std: tuple[float, float, float]
    forbidden_text_tokens: tuple[str, ...]
    video_fps = 2.0
    maximum_video_frames = 0

    def __init__(self, avfoundation_library: str | Path | None = None) -> None:
        library = avfoundation_library or os.environ.get("MFQ_AVFOUNDATION_VIDEO_LIBRARY")
        self._avfoundation_decoder: _AVFoundationVideoDecoder | None = None
        if platform.system() == "Darwin" and library and Path(library).is_file():
            try:
                self._avfoundation_decoder = _AVFoundationVideoDecoder(library)
            except OSError:
                self._avfoundation_decoder = None

    @classmethod
    def _normalize(cls, pixels: np.ndarray) -> np.ndarray:
        value = np.asarray(pixels, dtype=np.float32) / np.float32(255.0)
        mean = np.asarray(cls.image_mean, dtype=np.float32)[:, None, None]
        standard_deviation = np.asarray(cls.image_std, dtype=np.float32)[:, None, None]
        return np.ascontiguousarray((value - mean) / standard_deviation)

    @classmethod
    def _patchify(cls, image: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int]]:
        channel, height, width = image.shape
        patch = cls.patch_size
        merge = cls.merge_size
        grid_height = height // patch
        grid_width = width // patch
        if grid_height % merge or grid_width % merge:
            raise VisionProcessingError("Flash-Next image dimensions do not divide merge geometry")
        patches = image.reshape(
            channel,
            grid_height // merge,
            merge,
            patch,
            grid_width // merge,
            merge,
            patch,
        )
        patches = patches.transpose(1, 4, 2, 5, 0, 3, 6)
        patches = np.broadcast_to(
            patches[:, :, :, :, :, None, :, :],
            (*patches.shape[:5], cls.temporal_patch_size, *patches.shape[5:]),
        )
        flattened = patches.reshape(
            grid_height * grid_width,
            channel * cls.temporal_patch_size * patch * patch,
        )
        return (
            np.ascontiguousarray(flattened, dtype=np.float32),
            (1, grid_height, grid_width),
        )

    @classmethod
    def _patchify_video(
        cls,
        frames: np.ndarray,
    ) -> tuple[np.ndarray, tuple[int, int, int]]:
        if frames.ndim != 4:
            raise VisionProcessingError("Flash-Next video frames must have [T,C,H,W] shape")
        frame_count, channel, height, width = frames.shape
        patch = cls.patch_size
        temporal = cls.temporal_patch_size
        merge = cls.merge_size
        if frame_count <= 0:
            raise VisionProcessingError("Flash-Next video contains no frames")
        if padding := -frame_count % temporal:
            frames = np.concatenate(
                (frames, np.repeat(frames[-1:], padding, axis=0)),
                axis=0,
            )
            frame_count += padding
        grid_time = frame_count // temporal
        grid_height = height // patch
        grid_width = width // patch
        if grid_height % merge or grid_width % merge:
            raise VisionProcessingError("Flash-Next video dimensions do not divide merge geometry")
        patches = frames.reshape(
            grid_time,
            temporal,
            channel,
            grid_height // merge,
            merge,
            patch,
            grid_width // merge,
            merge,
            patch,
        )
        flattened = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(
            grid_time * grid_height * grid_width,
            channel * temporal * patch * patch,
        )
        return (
            np.ascontiguousarray(flattened, dtype=np.float32),
            (grid_time, grid_height, grid_width),
        )

    @classmethod
    def _prepare_image(cls, image: Any) -> tuple[np.ndarray, tuple[int, int, int]]:
        raise NotImplementedError

    @classmethod
    def _placeholder(cls, merged_tokens: int) -> str:
        raise NotImplementedError

    @classmethod
    def _sample_video_indices(
        cls,
        total_frames: int,
        frames_per_second: float,
        duration_seconds: float,
    ) -> tuple[int, ...]:
        raise NotImplementedError

    @classmethod
    def _prepare_video(
        cls,
        decoded: _DecodedVideo,
    ) -> tuple[np.ndarray, tuple[int, int, int], str]:
        raise NotImplementedError

    @classmethod
    def _decode_video_pyav(cls, data: bytes) -> _DecodedVideo:
        try:
            import av
        except ImportError as error:
            raise VisionProcessingError(
                "video input requires the optional PyAV dependency"
            ) from error

        try:
            with av.open(io.BytesIO(data), mode="r") as container:
                streams = [stream for stream in container.streams if stream.type == "video"]
                if not streams:
                    raise VisionProcessingError("video contains no video stream")
                stream = streams[0]
                stream.thread_type = "FRAME"
                source_fps = float(stream.average_rate) if stream.average_rate else 30.0
                if not math.isfinite(source_fps) or source_fps <= 0.0:
                    source_fps = 30.0
                if stream.duration is not None and stream.time_base is not None:
                    duration_seconds = float(stream.duration * stream.time_base)
                elif container.duration is not None:
                    duration_seconds = float(container.duration) / 1_000_000.0
                else:
                    duration_seconds = 0.0
                declared_frames = int(stream.frames or 0)
                if declared_frames > 0:
                    selected_indices = cls._sample_video_indices(
                        declared_frames,
                        source_fps,
                        duration_seconds,
                    )
                    selected_at: dict[int, list[int]] = {}
                    for output_index, source_index in enumerate(selected_indices):
                        selected_at.setdefault(int(source_index), []).append(output_index)
                    frames: list[_PreparedVideoFrame | None] = [None] * len(selected_indices)
                    last_selected = max(selected_at, default=-1)
                    for index, frame in enumerate(container.decode(stream)):
                        if index in selected_at:
                            image = frame.to_image().convert("RGB")
                            prepared = _PreparedVideoFrame(
                                image=image,
                                source_size=image.size,
                                presentation_seconds=(
                                    float(frame.time)
                                    if frame.time is not None
                                    else index / source_fps
                                ),
                            )
                            for output_index in selected_at[index]:
                                frames[output_index] = prepared
                        if index >= last_selected:
                            break
                    if any(frame is None for frame in frames):
                        raise VisionProcessingError(
                            "video ended before every selected frame was decoded"
                        )
                    if duration_seconds <= 0.0:
                        duration_seconds = declared_frames / source_fps
                    return _DecodedVideo(
                        frames=tuple(frame for frame in frames if frame is not None),
                        frame_indices=tuple(int(value) for value in selected_indices),
                        frames_per_second=source_fps,
                        duration_seconds=duration_seconds,
                    )

                all_frames: list[_PreparedVideoFrame] = []
                for index, frame in enumerate(container.decode(stream)):
                    image = frame.to_image().convert("RGB")
                    all_frames.append(
                        _PreparedVideoFrame(
                            image=image,
                            source_size=image.size,
                            presentation_seconds=(
                                float(frame.time)
                                if frame.time is not None
                                else index / source_fps
                            ),
                        )
                    )
                if not all_frames:
                    raise VisionProcessingError("video contains no decodable frames")
                if duration_seconds <= 0.0:
                    duration_seconds = len(all_frames) / source_fps
                selected_indices = cls._sample_video_indices(
                    len(all_frames),
                    source_fps,
                    duration_seconds,
                )
                return _DecodedVideo(
                    frames=tuple(all_frames[index] for index in selected_indices),
                    frame_indices=tuple(int(value) for value in selected_indices),
                    frames_per_second=source_fps,
                    duration_seconds=duration_seconds,
                )
        except VisionProcessingError:
            raise
        except Exception as error:
            raise VisionProcessingError(f"unable to decode video: {error}") from error

    def _decode_video_for_request(self, data: bytes) -> _DecodedVideo:
        decoder = self._avfoundation_decoder
        if decoder is not None:
            try:
                return decoder.decode_sampled(
                    data,
                    sample_indices=self._sample_video_indices,
                    resize=lambda image: image,
                )
            except (OSError, RuntimeError, VisionProcessingError, ValueError):
                pass
        return self._decode_video_pyav(data)

    @classmethod
    def _pack_tensors(
        cls,
        patches: list[np.ndarray],
        grids: list[tuple[int, int, int]],
        media_types: list[int],
        *,
        use_binary_file: bool,
    ) -> tuple[dict[str, Any], tuple[Path, ...]]:
        if not patches or len(patches) != len(grids) or len(grids) != len(media_types):
            raise VisionProcessingError("Flash-Next media tensor geometry is invalid")
        if any(value not in (1, 2) for value in media_types):
            raise VisionProcessingError("Flash-Next media type is invalid")
        image_grids = [
            grid
            for grid, media_type in zip(grids, media_types, strict=True)
            if media_type == 1
        ]
        video_grids = [
            grid
            for grid, media_type in zip(grids, media_types, strict=True)
            if media_type == 2
        ]
        values = [
            ("pixel_values", np.concatenate(patches, axis=0), "float32"),
            ("vision_grid_thw", np.asarray(grids, dtype=np.int32), "int32"),
            ("vision_types", np.asarray(media_types, dtype=np.int32), "int32"),
        ]
        if image_grids:
            values.append(
                ("image_grid_thw", np.asarray(image_grids, dtype=np.int32), "int32")
            )
        if video_grids:
            values.append(
                ("video_grid_thw", np.asarray(video_grids, dtype=np.int32), "int32")
            )
        if use_binary_file:
            tensors, path = _binary_tensors(values)
            tensors["version"] = 3
            tensors["processor"] = cls.input_contract
            return tensors, (path,)
        return (
            {
                "version": 3,
                "processor": cls.input_contract,
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
        grids: list[tuple[int, int, int]] = []
        media_types: list[int] = []
        sources = 0
        frame_count = 0
        for message in prepared:
            content = message.get("content")
            if isinstance(content, str):
                if any(token in content for token in self.forbidden_text_tokens):
                    raise VisionProcessingError(
                        "Flash-Next vision placeholder tokens cannot be supplied as text"
                    )
                continue
            if not isinstance(content, list):
                continue
            pieces: list[str] = []
            for item in content:
                if not isinstance(item, dict):
                    raise VisionProcessingError("multimodal content item must be an object")
                item_type = item.get("type")
                if item_type == "text":
                    value = str(item.get("text", ""))
                    if any(token in value for token in self.forbidden_text_tokens):
                        raise VisionProcessingError(
                            "Flash-Next vision placeholder tokens cannot be supplied as text"
                        )
                    pieces.append(value)
                    continue
                if item_type not in {"image_url", "video_url"}:
                    raise VisionProcessingError(
                        f"{self.processor_name} does not yet support {item_type or 'unknown'} input"
                    )
                if self.processor_name == "qwen4_exp" and message.get("role") == "system":
                    raise VisionProcessingError("Qwen4-Exp system messages cannot contain media")
                if item_type == "image_url":
                    image_spec = item.get("image_url")
                    if not isinstance(image_spec, dict) or not isinstance(
                        image_spec.get("url"), str
                    ):
                        raise VisionProcessingError("image_url content is missing its URL")
                    image = _decode_image(
                        _decode_data_url(
                            image_spec["url"],
                            "image/",
                        )
                    )
                    pixel_values, grid = self._prepare_image(image)
                    merged_tokens = math.prod(grid) // (self.merge_size**2)
                    placeholder = self._placeholder(merged_tokens)
                    media_type = 1
                else:
                    video_spec = item.get("video_url")
                    if not isinstance(video_spec, dict) or not isinstance(
                        video_spec.get("url"), str
                    ):
                        raise VisionProcessingError("video_url content is missing its URL")
                    decoded = self._decode_video_for_request(
                        _decode_data_url(
                            video_spec["url"],
                            "video/",
                        )
                    )
                    pixel_values, grid, placeholder = self._prepare_video(decoded)
                    media_type = 2
                    frame_count += len(decoded.frames)
                patches.append(pixel_values)
                grids.append(grid)
                media_types.append(media_type)
                pieces.append(placeholder)
                sources += 1
            message["content"] = "".join(pieces)
        if not patches:
            return None
        tensors, cleanup_paths = self._pack_tensors(
            patches,
            grids,
            media_types,
            use_binary_file=use_binary_file,
        )
        return ProcessedVisionRequest(
            messages=prepared,
            tensors=tensors,
            source_count=sources,
            frame_count=frame_count,
            cleanup_paths=cleanup_paths,
        )
