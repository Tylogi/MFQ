"""Video frame decoding shared by multimodal processors."""

from __future__ import annotations

import math
import os
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np
from pydantic.dataclasses import dataclass

from mfq.server.vision.decode import VisionProcessingError


@dataclass(frozen=True)
class _PreparedVideoFrame:
    image: Any
    source_size: tuple[int, int]
    presentation_seconds: float = 0.0


@dataclass(frozen=True)
class _DecodedVideo:
    frames: tuple[_PreparedVideoFrame, ...]
    frame_indices: tuple[int, ...]
    frames_per_second: float
    duration_seconds: float


class _AVFoundationVideoDecoder:
    def __init__(self, library: str | Path) -> None:
        import ctypes

        self._ctypes = ctypes
        self._callback_type = ctypes.CFUNCTYPE(
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_double,
        )
        self._library = ctypes.CDLL(str(library))
        function = self._library.mfq_avfoundation_sample_video
        function.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_int,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int,
            self._callback_type,
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        function.restype = ctypes.c_int
        self._sample = function

    def _decode_selected(
        self,
        data: bytes,
        *,
        select_indices: Any,
        resize: Any,
    ) -> _DecodedVideo:
        import av

        descriptor, temporary_path = tempfile.mkstemp(prefix="mfq-video-", suffix=".mp4")
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
            with av.open(temporary_path, mode="r") as container:
                streams = [stream for stream in container.streams if stream.type == "video"]
                if not streams:
                    raise VisionProcessingError("video contains no video stream")
                stream = streams[0]
                if stream.time_base is None:
                    raise VisionProcessingError("video stream has no time base")
                time_base = stream.time_base
                frames_per_second = (
                    float(stream.average_rate) if stream.average_rate else 30.0
                )
                if not math.isfinite(frames_per_second) or frames_per_second <= 0.0:
                    frames_per_second = 30.0
                if stream.duration is not None:
                    duration_seconds = float(stream.duration * time_base)
                elif container.duration is not None:
                    duration_seconds = float(container.duration) / 1_000_000.0
                else:
                    duration_seconds = 0.0
                codec = stream.codec_context
                color_metadata = {
                    name: getattr(codec, name)
                    for name in (
                        "color_range",
                        "colorspace",
                        "color_primaries",
                        "color_trc",
                    )
                }
                packet_pts = sorted(
                    packet.pts for packet in container.demux(stream) if packet.pts is not None
                )
            if not packet_pts:
                raise VisionProcessingError("video contains no decodable frames")
            if duration_seconds <= 0.0:
                duration_seconds = len(packet_pts) / frames_per_second
            selected_indices = tuple(
                int(value)
                for value in select_indices(
                    tuple(packet_pts),
                    time_base,
                    frames_per_second,
                    duration_seconds,
                )
            )
            if not selected_indices:
                raise VisionProcessingError("video sampling selected no frames")
            if any(value < 0 or value >= len(packet_pts) for value in selected_indices):
                raise VisionProcessingError("video sampling selected an invalid frame index")
            selected_pts = [packet_pts[value] for value in selected_indices]

            ctypes = self._ctypes
            targets = (ctypes.c_int64 * len(selected_pts))(*selected_pts)
            frames: list[_PreparedVideoFrame | None] = [None] * len(selected_pts)
            callback_error: list[BaseException] = []

            def receive(
                _context: Any,
                target_index: int,
                y_plane: Any,
                y_stride: int,
                _uv_plane: Any,
                uv_stride: int,
                width: int,
                height: int,
                presentation_seconds: float,
            ) -> int:
                try:
                    if target_index < 0 or target_index >= len(frames):
                        raise VisionProcessingError("AVFoundation returned an invalid frame index")
                    if y_stride != width or uv_stride != width:
                        raise VisionProcessingError(
                            "AVFoundation returned non-compact video planes"
                        )
                    raw = np.ctypeslib.as_array(
                        y_plane,
                        shape=(width * height * 3 // 2,),
                    ).reshape(height * 3 // 2, width)
                    frame = av.VideoFrame.from_ndarray(raw, format="nv12").reformat(
                        format="yuv420p"
                    )
                    for name, value in color_metadata.items():
                        setattr(frame, name, value)
                    image = frame.to_image().convert("RGB")
                    frames[target_index] = _PreparedVideoFrame(
                        image=resize(image),
                        source_size=image.size,
                        presentation_seconds=float(presentation_seconds),
                    )
                    return 0
                except BaseException as error:
                    callback_error.append(error)
                    return 1

            callback = self._callback_type(receive)
            error_message = ctypes.create_string_buffer(1024)
            configured_parallelism = int(os.environ.get("MFQ_AVFOUNDATION_VIDEO_PARALLELISM", "16"))
            parallelism = max(1, min(32, configured_parallelism, len(selected_pts)))
            result = self._sample(
                os.fsencode(temporary_path),
                targets,
                len(selected_pts),
                time_base.numerator,
                time_base.denominator,
                parallelism,
                callback,
                None,
                error_message,
                len(error_message),
            )
            if callback_error:
                raise callback_error[0]
            if result != len(selected_pts):
                message = error_message.value.decode("utf-8", errors="replace")
                raise VisionProcessingError(
                    message or "AVFoundation did not return every selected video frame"
                )
            if any(frame is None for frame in frames):
                raise VisionProcessingError("AVFoundation omitted a selected video frame")
            return _DecodedVideo(
                frames=tuple(frame for frame in frames if frame is not None),
                frame_indices=selected_indices,
                frames_per_second=frames_per_second,
                duration_seconds=duration_seconds,
            )
        finally:
            with suppress(FileNotFoundError):
                os.unlink(temporary_path)

    def decode(
        self,
        data: bytes,
        *,
        frames_per_second: float,
        maximum_frames: int,
        resize: Any,
    ) -> list[_PreparedVideoFrame]:
        def select(
            packet_pts: tuple[int, ...],
            time_base: Any,
            _source_fps: float,
            _duration_seconds: float,
        ) -> tuple[int, ...]:
            selected: list[int] = []
            next_timestamp: float | None = None
            for index, pts in enumerate(packet_pts):
                timestamp = float(pts * time_base)
                if next_timestamp is not None and timestamp + 1.0e-9 < next_timestamp:
                    continue
                selected.append(index)
                next_timestamp = timestamp + 1.0 / frames_per_second
                if len(selected) >= maximum_frames:
                    break
            return tuple(selected)

        return list(
            self._decode_selected(
                data,
                select_indices=select,
                resize=resize,
            ).frames
        )

    def decode_sampled(
        self,
        data: bytes,
        *,
        sample_indices: Any,
        resize: Any,
    ) -> _DecodedVideo:
        def select(
            packet_pts: tuple[int, ...],
            _time_base: Any,
            source_fps: float,
            duration_seconds: float,
        ) -> tuple[int, ...]:
            return tuple(
                int(value)
                for value in sample_indices(
                    len(packet_pts),
                    source_fps,
                    duration_seconds,
                )
            )

        return self._decode_selected(
            data,
            select_indices=select,
            resize=resize,
        )
