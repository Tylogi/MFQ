"""MiniCPM-o 4.5 multimodal processor (audio, image, and video)."""

from __future__ import annotations

import base64
import copy
import io
import math
import os
import platform
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
from mfq.server.vision.video import (
    _AVFoundationVideoDecoder,
    _PreparedVideoFrame,
)


class MiniCPMO45VisionProcessor:
    """Exact CPU port of the official MiniCPM-o 4.5 media processors."""

    patch_size = 14
    scale_resolution = 448
    image_feature_size = 64
    maximum_image_slices = 9
    maximum_video_frames = 64
    video_fps = 1.0
    audio_sample_rate = 16_000
    audio_chunk_samples = 30 * audio_sample_rate
    minimum_audio_samples = audio_sample_rate // 10
    maximum_audio_samples = 30 * 60 * audio_sample_rate

    def __init__(self, avfoundation_library: str | Path | None = None) -> None:
        library = avfoundation_library or os.environ.get("MFQ_AVFOUNDATION_VIDEO_LIBRARY")
        self._avfoundation_decoder: _AVFoundationVideoDecoder | None = None
        self._mel_extractor: Any | None = None
        if platform.system() == "Darwin" and library and Path(library).is_file():
            try:
                self._avfoundation_decoder = _AVFoundationVideoDecoder(library)
            except (OSError, AttributeError):
                self._avfoundation_decoder = None

    @staticmethod
    def _ensure_divide(length: int, divisor: int) -> int:
        return max(round(length / divisor) * divisor, divisor)

    @classmethod
    def _best_resize(
        cls,
        size: tuple[int, int],
        *,
        allow_upscale: bool,
    ) -> tuple[int, int]:
        width, height = size
        if width <= 0 or height <= 0:
            raise VisionProcessingError("image dimensions must be positive")
        if width * height > cls.scale_resolution**2 or allow_upscale:
            ratio = width / height
            height = int(cls.scale_resolution / math.sqrt(ratio))
            width = int(height * ratio)
        return (
            cls._ensure_divide(width, cls.patch_size),
            cls._ensure_divide(height, cls.patch_size),
        )

    @classmethod
    def _sliced_grid(
        cls,
        size: tuple[int, int],
        maximum_slices: int,
    ) -> tuple[int, int] | None:
        width, height = size
        ratio = width * height / cls.scale_resolution**2
        multiple = min(math.ceil(ratio), maximum_slices)
        if multiple <= 1:
            return None
        candidates: list[tuple[int, int]] = []
        for count in (multiple - 1, multiple, multiple + 1):
            if count == 1 or count > maximum_slices:
                continue
            for columns in range(1, count + 1):
                if count % columns == 0:
                    candidates.append((columns, count // columns))
        if not candidates:
            return None
        log_ratio = math.log(width / height)
        return min(candidates, key=lambda grid: abs(log_ratio - math.log(grid[0] / grid[1])))

    @classmethod
    def _refine_size(
        cls,
        size: tuple[int, int],
        grid: tuple[int, int],
    ) -> tuple[int, int]:
        width, height = size
        columns, rows = grid
        refined_width = cls._ensure_divide(width, columns)
        refined_height = cls._ensure_divide(height, rows)
        tile = cls._best_resize(
            (refined_width / columns, refined_height / rows),
            allow_upscale=True,
        )
        return tile[0] * columns, tile[1] * rows

    @classmethod
    def _slice_image(cls, image: Any, maximum_slices: int) -> tuple[list[Any], Any]:
        from PIL import Image

        image = image.convert("RGB")
        grid = cls._sliced_grid(image.size, maximum_slices)
        overview = image.resize(
            cls._best_resize(image.size, allow_upscale=grid is None),
            resample=Image.Resampling.BICUBIC,
        )
        slices = [overview]
        if grid is not None:
            refined = image.resize(
                cls._refine_size(image.size, grid),
                resample=Image.Resampling.BICUBIC,
            )
            columns, rows = grid
            tile_width = refined.width // columns
            tile_height = refined.height // rows
            for row in range(rows):
                for column in range(columns):
                    slices.append(
                        refined.crop(
                            (
                                column * tile_width,
                                row * tile_height,
                                (column + 1) * tile_width,
                                (row + 1) * tile_height,
                            )
                        )
                    )
        return slices, grid

    @classmethod
    def _reshape_by_patch(cls, image: Any) -> tuple[np.ndarray, tuple[int, int]]:
        value = np.asarray(image, dtype=np.float32) / np.float32(255.0)
        value = (value - np.float32(0.5)) / np.float32(0.5)
        value = np.transpose(value, (2, 0, 1))
        channels, height, width = value.shape
        patch = cls.patch_size
        if height % patch or width % patch:
            raise VisionProcessingError("processed image is not divisible by the patch size")
        rows = height // patch
        columns = width // patch
        packed = (
            value.reshape(channels, rows, patch, columns, patch)
            .transpose(0, 2, 1, 3, 4)
            .reshape(channels, patch, rows * columns * patch)
        )
        return np.ascontiguousarray(packed, dtype=np.float32), (rows, columns)

    @classmethod
    def _placeholder(
        cls,
        size: tuple[int, int],
        image_index: int,
        maximum_slices: int,
    ) -> str:
        unknowns = "<unk>" * cls.image_feature_size
        result = f"<image_id>{image_index}</image_id><image>{unknowns}</image>"
        grid = cls._sliced_grid(size, maximum_slices)
        if grid is None:
            return result
        columns, rows = grid
        slice_placeholder = f"<slice>{unknowns}</slice>"
        return result + "\n".join(slice_placeholder * columns for _ in range(rows))

    @staticmethod
    def _decode_audio_data(value: Any) -> bytes:
        if not isinstance(value, dict) or not isinstance(value.get("data"), str):
            raise VisionProcessingError("input_audio content is missing its base64 data")
        try:
            data = base64.b64decode(value["data"], validate=True)
        except ValueError as error:
            raise VisionProcessingError("audio input contains invalid base64") from error
        if not data:
            raise VisionProcessingError("audio input is empty")
        return data

    @classmethod
    def _decode_audio(cls, data: bytes) -> np.ndarray:
        try:
            import av
        except ImportError:
            # The standard library covers ordinary uncompressed WAV input,
            # which keeps the native audio path useful without PyAV. Other
            # containers still use PyAV for demuxing and resampling.
            try:
                import wave

                with wave.open(io.BytesIO(data), "rb") as source:
                    channels = source.getnchannels()
                    sample_width = source.getsampwidth()
                    sample_rate = source.getframerate()
                    frame_count = source.getnframes()
                    compression = source.getcomptype()
                    if channels <= 0 or sample_rate <= 0 or frame_count <= 0:
                        raise VisionProcessingError("audio WAV geometry is invalid")
                    if compression != "NONE" or sample_width not in {1, 2, 3, 4}:
                        raise VisionProcessingError(
                            "audio WAV encoding requires the optional PyAV dependency"
                        )
                    target_count = int(round(
                        frame_count * cls.audio_sample_rate / sample_rate
                    ))
                    if target_count > cls.maximum_audio_samples:
                        raise VisionProcessingError("audio input exceeds the 30 minute limit")
                    payload = source.readframes(frame_count)
            except VisionProcessingError:
                raise
            except Exception as error:
                raise VisionProcessingError(
                    "audio input requires PyAV or an uncompressed PCM WAV"
                ) from error

            if sample_width == 1:
                waveform = (
                    np.frombuffer(payload, dtype=np.uint8).astype(np.float32) - 128.0
                ) / 128.0
            elif sample_width == 2:
                waveform = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
            elif sample_width == 3:
                packed = np.frombuffer(payload, dtype=np.uint8).reshape(-1, 3)
                values = (
                    packed[:, 0].astype(np.int32)
                    | (packed[:, 1].astype(np.int32) << 8)
                    | (packed[:, 2].astype(np.int32) << 16)
                )
                values = (values ^ 0x800000) - 0x800000
                waveform = values.astype(np.float32) / 8388608.0
            else:
                waveform = np.frombuffer(payload, dtype="<i4").astype(np.float32) / 2147483648.0
            if waveform.size != frame_count * channels:
                raise VisionProcessingError("audio WAV payload is truncated") from None
            waveform = waveform.reshape(frame_count, channels).mean(axis=1)
            if sample_rate != cls.audio_sample_rate:
                source_positions = np.arange(frame_count, dtype=np.float64)
                target_positions = np.arange(target_count, dtype=np.float64) * (
                    sample_rate / cls.audio_sample_rate
                )
                waveform = np.interp(
                    target_positions, source_positions, waveform
                ).astype(np.float32)
            waveform = np.ascontiguousarray(waveform, dtype=np.float32)
            if not np.isfinite(waveform).all():
                raise VisionProcessingError("audio input contains a non-finite sample") from None
            return np.clip(waveform, -1.0, 1.0)

        chunks: list[np.ndarray] = []
        try:
            with av.open(io.BytesIO(data), mode="r") as container:
                streams = [stream for stream in container.streams if stream.type == "audio"]
                if not streams:
                    raise VisionProcessingError("audio contains no audio stream")
                resampler = av.AudioResampler(
                    format="fltp",
                    layout="mono",
                    rate=cls.audio_sample_rate,
                )

                def append_frames(frames: Any) -> None:
                    if frames is None:
                        return
                    if not isinstance(frames, list):
                        frames = [frames]
                    for frame in frames:
                        if frame is not None:
                            chunks.append(
                                np.ascontiguousarray(
                                    frame.to_ndarray().reshape(-1),
                                    dtype=np.float32,
                                )
                            )

                for frame in container.decode(streams[0]):
                    append_frames(resampler.resample(frame))
                append_frames(resampler.resample(None))
        except VisionProcessingError:
            raise
        except Exception as error:
            raise VisionProcessingError(f"unable to decode audio: {error}") from error
        if not chunks:
            raise VisionProcessingError("audio contains no decodable samples")
        waveform = np.ascontiguousarray(np.concatenate(chunks), dtype=np.float32)
        if waveform.size > cls.maximum_audio_samples:
            raise VisionProcessingError("audio input exceeds the 30 minute limit")
        if not np.isfinite(waveform).all():
            raise VisionProcessingError("audio input contains a non-finite sample")
        return np.clip(waveform, -1.0, 1.0)

    @staticmethod
    def _audio_placeholder(frame_count: int) -> str:
        after_convolution = (frame_count - 1) // 2 + 1
        pooled = (after_convolution - 5) // 5 + 1
        if frame_count <= 0 or pooled <= 0:
            raise VisionProcessingError("audio input is too short")
        return f"<|audio_start|>{'<unk>' * pooled}<|audio_end|>"

    def _prepare_audio(self, data: bytes) -> list[np.ndarray]:
        from mfq.runtime.minicpmo45_realtime import MiniCPMOMel

        waveform = self._decode_audio(data)
        if waveform.size < self.minimum_audio_samples:
            waveform = np.pad(
                waveform,
                (0, self.minimum_audio_samples - waveform.size),
            )
        if self._mel_extractor is None:
            self._mel_extractor = MiniCPMOMel()
        result: list[np.ndarray] = []
        for start in range(0, waveform.size, self.audio_chunk_samples):
            chunk = waveform[start : start + self.audio_chunk_samples]
            if chunk.size < self.minimum_audio_samples:
                chunk = np.pad(chunk, (0, self.minimum_audio_samples - chunk.size))
            features = self._mel_extractor.extract(chunk, fixed_floor=False)
            if features.shape[0] != 80 or not 9 <= features.shape[1] <= 3000:
                raise VisionProcessingError("processed audio tensor geometry is invalid")
            result.append(np.ascontiguousarray(features, dtype=np.float32))
        return result

    @classmethod
    def _decode_video(cls, data: bytes) -> list[Any]:
        try:
            import av
        except ImportError as error:
            raise VisionProcessingError(
                "video input requires the optional PyAV dependency"
            ) from error

        def open_stream() -> tuple[Any, Any]:
            container = av.open(io.BytesIO(data), mode="r")
            streams = [stream for stream in container.streams if stream.type == "video"]
            if not streams:
                container.close()
                raise VisionProcessingError("video contains no video stream")
            stream = streams[0]
            # PyAV defaults to slice-only decoding. Frame threading makes
            # inter-frame codecs use the available CPU cores without changing
            # decoded pixels or presentation timestamps.
            stream.thread_type = "FRAME"
            return container, stream

        def timestamp_of(frame: Any, index: int, average_rate: float) -> float:
            if frame.time is not None:
                return float(frame.time)
            if frame.pts is not None and frame.time_base is not None:
                return float(frame.pts * frame.time_base)
            return index / max(average_rate, 1.0)

        def decode_sequentially() -> list[Any]:
            selected: list[Any] = []
            next_timestamp = 0.0
            container, stream = open_stream()
            try:
                average_rate = float(stream.average_rate) if stream.average_rate else 30.0
                for index, frame in enumerate(container.decode(stream)):
                    timestamp = timestamp_of(frame, index, average_rate)
                    if selected and timestamp + 1.0e-9 < next_timestamp:
                        continue
                    selected.append(frame.to_image().convert("RGB"))
                    next_timestamp = timestamp + 1.0 / cls.video_fps
                    if len(selected) >= cls.maximum_video_frames:
                        break
            finally:
                container.close()
            return selected

        frames: list[Any] = []
        try:
            try:
                # Indexed containers can jump to the keyframe preceding each
                # requested timestamp. This avoids decoding long spans that
                # will never be sampled. If seeking is unavailable, the exact
                # sequential path remains the compatibility fallback.
                container, stream = open_stream()
                try:
                    if stream.time_base is None:
                        raise RuntimeError("video stream has no seek time base")
                    time_base = float(stream.time_base)
                    average_rate = float(stream.average_rate) if stream.average_rate else 30.0
                    next_timestamp = 0.0
                    while len(frames) < cls.maximum_video_frames:
                        container.seek(
                            max(0, int(next_timestamp / time_base)),
                            stream=stream,
                            backward=True,
                            any_frame=False,
                        )
                        selected = None
                        for index, frame in enumerate(container.decode(stream)):
                            timestamp = timestamp_of(frame, index, average_rate)
                            if timestamp + 1.0e-9 >= next_timestamp:
                                selected = (frame, timestamp)
                                break
                        if selected is None:
                            break
                        frame, timestamp = selected
                        frames.append(frame.to_image().convert("RGB"))
                        next_timestamp = timestamp + 1.0 / cls.video_fps
                finally:
                    container.close()
            except VisionProcessingError:
                raise
            except Exception:
                frames = decode_sequentially()
            if not frames:
                frames = decode_sequentially()
        except VisionProcessingError:
            raise
        except Exception as error:
            raise VisionProcessingError(f"unable to decode video: {error}") from error
        if not frames:
            raise VisionProcessingError("video contains no decodable frames")
        return frames

    @classmethod
    def _resize_video_frame(cls, image: Any) -> Any:
        from PIL import Image

        return image.resize(
            cls._best_resize(image.size, allow_upscale=True),
            resample=Image.Resampling.BICUBIC,
        )

    def _decode_video_for_request(self, data: bytes) -> list[_PreparedVideoFrame]:
        decoder = self._avfoundation_decoder
        if decoder is not None:
            try:
                return decoder.decode(
                    data,
                    frames_per_second=self.video_fps,
                    maximum_frames=self.maximum_video_frames,
                    resize=self._resize_video_frame,
                )
            except (OSError, RuntimeError, VisionProcessingError, ValueError):
                pass
        return [
            _PreparedVideoFrame(
                image=self._resize_video_frame(image),
                source_size=image.size,
            )
            for image in self._decode_video(data)
        ]

    @classmethod
    def _pack_tensors(
        cls,
        patches: list[np.ndarray],
        target_sizes: list[tuple[int, int]],
        audio_features: list[np.ndarray],
        *,
        use_binary_file: bool = False,
    ) -> tuple[dict[str, Any], tuple[Path, ...]]:
        values: list[tuple[str, np.ndarray, str]] = []
        if patches or target_sizes:
            if not patches or len(patches) != len(target_sizes):
                raise VisionProcessingError("processed image tensor geometry is invalid")
            sequences = [patch.reshape(3 * cls.patch_size, -1).T for patch in patches]
            maximum_length = max(sequence.shape[0] for sequence in sequences)
            padded = np.zeros(
                (len(sequences), maximum_length, 3 * cls.patch_size),
                dtype=np.float32,
            )
            for index, sequence in enumerate(sequences):
                padded[index, : sequence.shape[0]] = sequence
            pixels = padded.transpose(0, 2, 1).reshape(
                len(sequences), 3, cls.patch_size, maximum_length
            )
            sizes = np.asarray(target_sizes, dtype=np.int32)
            maximum_patches = int(np.max(sizes[:, 0] * sizes[:, 1]))
            mask = np.zeros((len(sequences), maximum_patches), dtype=np.uint8)
            for index, (rows, columns) in enumerate(target_sizes):
                mask[index, : rows * columns] = 1
            values.extend(
                [
                    ("pixel_values", pixels, "float32"),
                    ("patch_mask", mask, "uint8"),
                    ("target_sizes", sizes, "int32"),
                ]
            )
        if audio_features:
            maximum_frames = max(features.shape[1] for features in audio_features)
            padded_audio = np.zeros(
                (len(audio_features), 80, maximum_frames),
                dtype=np.float32,
            )
            lengths = np.empty(len(audio_features), dtype=np.int64)
            for index, features in enumerate(audio_features):
                if features.ndim != 2 or features.shape[0] != 80:
                    raise VisionProcessingError("processed audio tensor geometry is invalid")
                padded_audio[index, :, : features.shape[1]] = features
                lengths[index] = features.shape[1]
            values.extend(
                [
                    ("audio_features", padded_audio, "float32"),
                    ("audio_lengths", lengths, "int64"),
                ]
            )
        if not values:
            raise VisionProcessingError("multimodal request contains no processed media")
        if use_binary_file:
            tensors, path = _binary_tensors(values)
            return tensors, (path,)
        return (
            {
                "version": 1,
                **{name: _tensor(value, dtype) for name, value, dtype in values},
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
        target_sizes: list[tuple[int, int]] = []
        audio_features: list[np.ndarray] = []
        source_count = 0
        frame_count = 0
        image_index = 0
        found_media = False
        for message in prepared:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            pieces: list[str] = []
            for item in content:
                if not isinstance(item, dict):
                    raise VisionProcessingError("multimodal content item must be an object")
                item_type = item.get("type")
                if item_type == "text":
                    pieces.append(str(item.get("text", "")))
                    continue
                if item_type == "image_url":
                    image_spec = item.get("image_url")
                    if not isinstance(image_spec, dict) or not isinstance(
                        image_spec.get("url"), str
                    ):
                        raise VisionProcessingError("image_url content is missing its URL")
                    images = [
                        _decode_image(_decode_data_url(image_spec["url"], "image/"))
                    ]
                    maximum_slices = self.maximum_image_slices
                    source_count += 1
                elif item_type == "video_url":
                    video_spec = item.get("video_url")
                    if not isinstance(video_spec, dict) or not isinstance(
                        video_spec.get("url"), str
                    ):
                        raise VisionProcessingError("video_url content is missing its URL")
                    video_frames = self._decode_video_for_request(
                        _decode_data_url(video_spec["url"], "video/")
                    )
                    images = [frame.image for frame in video_frames]
                    maximum_slices = 1
                    source_count += 1
                    frame_count += len(images)
                elif item_type == "input_audio":
                    features_list = self._prepare_audio(
                        self._decode_audio_data(item.get("input_audio"))
                    )
                    source_count += 1
                    found_media = True
                    for features in features_list:
                        pieces.append(self._audio_placeholder(features.shape[1]))
                        audio_features.append(features)
                    continue
                else:
                    raise VisionProcessingError(f"unsupported multimodal content type: {item_type}")
                found_media = True
                for media_index, image in enumerate(images):
                    source_size = (
                        video_frames[media_index].source_size
                        if item_type == "video_url"
                        else image.size
                    )
                    pieces.append(self._placeholder(source_size, image_index, maximum_slices))
                    sliced = (
                        [image]
                        if item_type == "video_url"
                        else self._slice_image(image, maximum_slices)[0]
                    )
                    for image_slice in sliced:
                        packed, target = self._reshape_by_patch(image_slice)
                        patches.append(packed)
                        target_sizes.append(target)
                    image_index += 1
            message["content"] = "\n".join(piece for piece in pieces if piece)
        if not found_media:
            return None
        tensors, cleanup_paths = self._pack_tensors(
            patches,
            target_sizes,
            audio_features,
            use_binary_file=use_binary_file,
        )
        return ProcessedVisionRequest(
            messages=prepared,
            tensors=tensors,
            source_count=source_count,
            frame_count=frame_count,
            cleanup_paths=cleanup_paths,
        )
