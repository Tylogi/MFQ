from __future__ import annotations

import json
from typing import Any

from mfq.formats.assets import MODEL_CONFIG_ASSET
from mfq.formats.io import open_mmap
from mfq.server.services.model_context import yarn_context_limits
from mfq.server.state.catalog import DiscoveredModel

QSA_KV_BUFFER_BYTES = 64 << 20


def qsa_index_requirement(artifact: DiscoveredModel, context: int, budget_bytes: int | None = None,
    *, maximum_context: int | None = None, cuda: bool = False) -> int:
    with open_mmap(artifact.path) as store:
        record = store.records.get(MODEL_CONFIG_ASSET)
        if record is None or record.nbytes > 4 << 20:
            raise ValueError("QSA model configuration is unavailable")
        config = json.loads(store.read_blob(MODEL_CONFIG_ASSET))
        text = config.get("text_config", config)
        if not isinstance(text, dict) or text.get("model_type") not in {"qwen4_exp", "qwen4_exp_text"}:
            raise ValueError("KV offload currently supports QSA only")

        def integer(name: str, default: int | None = None) -> int:
            value: Any = text.get(name, default)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"invalid QSA {name}")
            return value

        maximum = integer("max_position_embeddings")
        native, extended, _ = yarn_context_limits(config, artifact.resource.architecture)
        if maximum_context is not None:
            if maximum_context > (extended or native or maximum):
                raise ValueError("QSA target context exceeds the YaRN extension limit")
            maximum = maximum_context
        elif native is not None:
            maximum = native
        count = integer("num_hidden_layers")
        width = integer("indexer_head_dim")
        ratio = integer("indexer_compress_ratio")
        interval = integer("full_attention_interval", 4)
        if min(maximum, count, width, ratio, interval) <= 0 or context < 512 or context > maximum:
            raise ValueError("QSA target context exceeds the model capacity")
        types = text.get("layer_types", ["full_attention" if (i + 1) % interval == 0 else "linear_attention" for i in range(count)])
        if not isinstance(types, list) or len(types) != count:
            raise ValueError("invalid QSA layer schedule")
        layers = types.count("full_attention")
        if any(name.startswith("predictor.block.") for name in store.records):
            layers += integer("mtp_num_hidden_layers", 0)
        if not layers:
            raise ValueError("the model has no QSA layers")
        if budget_bytes is not None:
            heads, dimension = integer("num_key_value_heads"), integer("head_dim")
            buffer = max(4096, heads * dimension * 4 * ratio * 4, width * 16 * 4 * 2)
            if min(heads, dimension) <= 0 or min(QSA_KV_BUFFER_BYTES, budget_bytes // 4) < buffer:
                raise ValueError(f"QSA KV budget needs at least {buffer * 4} bytes for microblock I/O; this is not an indexer residency requirement")
            if cuda:
                selected = min(context, integer("indexer_budget", 2048) + ratio - 1)
                staging = selected * heads * dimension * 4 * 4
                if min(QSA_KV_BUFFER_BYTES, budget_bytes // 4) < staging:
                    raise ValueError(f"CUDA QSA KV budget needs at least {staging * 4} bytes for one selected context")
        if cuda:
            return layers * width * (context + context // ratio) * 2
        return layers * (width * ((ratio + 5) * 2 + (context // ratio) * 4) + 2)
