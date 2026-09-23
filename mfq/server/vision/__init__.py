"""Architecture-neutral media preprocessing for native MFQ workers.

The wire format is deliberately shared while preprocessing remains owned by
the model family.  A processor may replace media blocks with prompt markers
and attach named tensors; the C++ runtime performs tokenizer-position
dependent expansion (for example DeepSeek-V4's N-layout image sentinels).
"""

from __future__ import annotations

from pathlib import Path

from mfq.server.vision.decode import (
    MultimodalProcessor,
    ProcessedVisionRequest,
    VisionProcessingError,
    clear_image_decode_cache,
)
from mfq.server.vision.deepseek import (
    DeepseekV4VisionProcessor,
    DeepseekV41VisionProcessor,
)
from mfq.server.vision.glm5_next import Glm5NextVisionProcessor
from mfq.server.vision.minicpm import MiniCPMO45VisionProcessor
from mfq.server.vision.qwen import Qwen4ExpVisionProcessor, Qwen35VisionProcessor
from mfq.server.vision.video import _DecodedVideo as _DecodedVideo
from mfq.server.vision.video import _PreparedVideoFrame as _PreparedVideoFrame


def multimodal_processor_for_architecture(
    model_type: str,
    *,
    avfoundation_library: str | Path | None = None,
) -> MultimodalProcessor | None:
    """Create the media processor registered for a runtime architecture."""

    identity = "_".join(
        part for part in model_type.strip().lower().replace("-", "_").split("_") if part
    )
    if identity == "minicpmo":
        return MiniCPMO45VisionProcessor(avfoundation_library=avfoundation_library)
    if identity in {
        "deepseek_v4",
        "deepseekv4",
        "deepseek_v4_vision",
    }:
        return DeepseekV4VisionProcessor()
    if identity in {
        "deepseek_v41",
        "deepseekv41",
        "deepseek_v41_text",
        "deepseek_v41_vision",
    }:
        return DeepseekV41VisionProcessor()
    if identity in {"qwen4_exp", "qwen4_exp_text"}:
        return Qwen4ExpVisionProcessor(avfoundation_library=avfoundation_library)
    if identity in {"qwen3_5", "qwen3_5_text", "qwen35"}:
        return Qwen35VisionProcessor(avfoundation_library=avfoundation_library)
    if identity in {"glm5_next", "glm5_next_text"}:
        return Glm5NextVisionProcessor(avfoundation_library=avfoundation_library)
    return None


__all__ = [
    "clear_image_decode_cache",
    "DeepseekV41VisionProcessor",
    "DeepseekV4VisionProcessor",
    "Glm5NextVisionProcessor",
    "MiniCPMO45VisionProcessor",
    "MultimodalProcessor",
    "ProcessedVisionRequest",
    "Qwen4ExpVisionProcessor",
    "Qwen35VisionProcessor",
    "VisionProcessingError",
    "multimodal_processor_for_architecture",
]
