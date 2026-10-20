"""Normalized architecture contracts shared by converters and runtimes."""

from mfq.architectures.deepseek_v41 import (
    DeepseekV41Config,
    DeepseekV41VisionConfig,
    parse_deepseek_v41_config,
)
from mfq.architectures.flash_next import (
    FlashNextConfig,
    Glm5NextConfig,
    Qwen4ExpConfig,
    VisionTowerConfig,
    parse_flash_next_config,
)
from mfq.architectures.nemotron_h import (
    NemotronHConfig,
    RadioProjectorConfig,
    RadioVisionConfig,
    parse_nemotron_h_config,
)

__all__ = [
    "DeepseekV41Config",
    "DeepseekV41VisionConfig",
    "FlashNextConfig",
    "Glm5NextConfig",
    "NemotronHConfig",
    "Qwen4ExpConfig",
    "RadioProjectorConfig",
    "RadioVisionConfig",
    "VisionTowerConfig",
    "parse_deepseek_v41_config",
    "parse_flash_next_config",
    "parse_nemotron_h_config",
]
