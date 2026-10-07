from __future__ import annotations

import json
import math
import re
import struct
from collections import defaultdict
from typing import Any

import numpy as np

from mfq.architectures.tensor_schema import graph_spec_for_source_names, map_source_tensor_name
from mfq.formats.assets import MODEL_CONFIG_ASSET, MODEL_GRAPH_ASSET, is_asset_record
from mfq.formats.inspection import inspect_tensor
from mfq.formats.io import open_mmap, view_mfe_blob
from mfq.server.protocol.analysis import (
    AnalysisAdvanced,
    AnalysisExpert,
    AnalysisLayer,
    AnalysisNormalization,
    AnalysisProjection,
    AnalysisTensor,
    CheckpointAnalysis,
)
from mfq.server.state.catalog import DiscoveredModel, ModelCatalog, _is_always_streamed_tensor
from mfq.server.services.model_memory import cache_profile

_LAYER = re.compile(r"^model\.block\.(\d+)\.")
_EXPERT = re.compile(r"\.experts\.(?:(\d+)\.)?(gate_up|gate|up|down)(?:_proj)?(?:\.weight)?$")


def _integer(config: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = config.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def _advanced(text: dict[str, Any], graph: dict[str, Any], tensors: list[AnalysisTensor]) -> AnalysisAdvanced:
    def string(*keys: str) -> str | None:
        return next((text[key].strip() for key in keys if isinstance(text.get(key), str) and text[key].strip()), None)

    def number(config: dict[str, Any], *keys: str) -> float | None:
        return next((float(config[key]) for key in keys if isinstance(config.get(key), (int, float))
            and not isinstance(config[key], bool) and math.isfinite(config[key]) and config[key] > 0), None)

    def width(category: str) -> int | None:
        for tensor in tensors:
            projection = tensor.projection.removesuffix("_proj")
            if tensor.category == category and projection in {"gate", "up", "gate_up"} and len(tensor.shape) >= 2:
                rows = tensor.shape[-2]
                if projection != "gate_up" or rows % 2 == 0:
                    return rows // (2 if projection == "gate_up" else 1)
        return None

    shared = [tensor for tensor in tensors if tensor.category == "shared_expert"]
    shared_count = _integer(text, "n_shared_experts", "num_shared_experts") if shared else 0
    if shared and shared_count is None:
        shared_count = next((tensor.shape[-3] for tensor in shared if len(tensor.shape) >= 3), 1)
    shared_width = _integer(text, "shared_expert_intermediate_size") or width("shared_expert")
    if shared_width and shared_count:
        shared_width = shared_width // shared_count if shared_width % shared_count == 0 else None
    hc_count = _integer(text, "hc_count", "hc_mult")
    architecture = str(text.get("model_type") or graph.get("architecture") or "")
    qwen4 = architecture in {"qwen4_exp", "qwen4_exp_text"}
    qwen35 = architecture in {"qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text", "qwen3_next"}
    has_gdn = any(tensor.category == "gdn" for tensor in tensors)
    has_ple = any(tensor.category in {"ple", "ple_projection"} for tensor in tensors)
    has_qsa = any(".attention.indexer." in tensor.name for tensor in tensors)
    has_full_gate = any(tensor.category == "attention" and (tensor.projection == "gate" or
        tensor.projection == "query" and len(tensor.shape) >= 2 and _integer(text, "num_attention_heads") and
        _integer(text, "head_dim") and tensor.shape[-2] == 2 * text["num_attention_heads"] * text["head_dim"]) for tensor in tensors)
    stride = _integer(text, "indexer_compress_ratio")
    budget = _integer(text, "indexer_budget")
    ratios = text.get("compress_ratios", [])
    rope = text.get("rope_parameters", {})
    if not isinstance(rope, dict):
        rope = {}
    has_predictor = any(tensor.category == "predictor" for tensor in tensors)
    topology = graph.get("topology", {})
    norm_eps = number(text, "rms_norm_eps", "norm_eps", "rms_eps")
    norms: dict[tuple[Any, ...], AnalysisNormalization] = {}
    sparse_layers = {tensor.layer for tensor in tensors if ".attention.indexer." in tensor.name}
    gdn_layers = {tensor.layer for tensor in tensors if tensor.category == "gdn"}
    tensor_names = {tensor.name for tensor in tensors}
    known_rms = qwen4 or qwen35 or architecture.startswith("deepseek") or architecture in {"qwen2", "qwen2_moe", "qwen3", "qwen3_moe", "llama", "mistral", "mixtral"}

    def norm(component: str, position: str, kind: str | None, dimension: int | None,
        epsilon: float | None, groups: int | None, layer: int | None) -> None:
        key = (component, position, kind, dimension, epsilon, groups)
        if key not in norms:
            norms[key] = AnalysisNormalization(component=component, position=position, kind=kind,
                dimension=dimension, epsilon=epsilon, groups=groups)
        if layer is not None and layer not in norms[key].layers:
            norms[key].layers.append(layer)

    for tensor in tensors:
        if not re.search(r"(?:^|\.)(?:\w*norm)\.weight$", tensor.name) or len(tensor.shape) != 1:
            continue
        component, position = "other", tensor.name.removesuffix(".weight")
        kind, groups, dimension = "RMSNorm" if known_rms else None, None, tensor.shape[0]
        epsilon = norm_eps
        name = tensor.name
        if ".mhc." in name:
            component = "residual"
            position = "ffn_input" if ".mlp." in name else "attention_input" if ".attention." in name else "final_readout"
            if qwen4 and hc_count and _integer(text, "hidden_size") and dimension == hc_count * text["hidden_size"]:
                kind, groups, dimension = "GroupedRMSNorm", hc_count, text["hidden_size"]
        elif tensor.category == "gdn":
            component, position = "gdn", "output"
        elif tensor.category == "attention":
            component = "indexer" if ".indexer." in name else "qsa" if tensor.layer in sparse_layers and qwen4 else "attention"
            position = name.split(".attention.", 1)[-1].removeprefix("indexer.").removesuffix(".weight").removesuffix("_norm")
            position = "input" if position == "norm" else position
            if component == "attention" and position == "input" and tensor.layer in gdn_layers:
                component = "gdn"
            if component == "indexer" and not qwen4:
                kind = "LayerNorm" if name.removesuffix(".weight") + ".bias" in tensor_names else None
                epsilon = number(text, "layer_norm_eps") if kind == "LayerNorm" else None
        elif tensor.category == "ple_projection":
            component = "ple"
            position = name.rsplit(".", 2)[-2].removesuffix("_norm")
            if qwen4 and hc_count and _integer(text, "hidden_size") and dimension == hc_count * text["hidden_size"]:
                kind, groups, dimension = "GroupedRMSNorm", hc_count, text["hidden_size"]
        elif tensor.category == "ffn":
            component, position = "ffn", "input"
        elif name == "model.output_norm.weight":
            component, position = "output", "output"
        if qwen4 or qwen35:
            head_norm = re.search(r"\.attention\.(indexer\.)?(query|key)_norm\.weight$", name)
            if head_norm:
                kind = "GroupedRMSNorm"
                if head_norm[1]:
                    keys = ("indexer_n_heads", "index_n_heads") if head_norm[2] == "query" else ("indexer_kv_heads",)
                else:
                    keys = ("num_attention_heads",) if head_norm[2] == "query" else ("num_key_value_heads",)
                groups = _integer(text, *keys)
            elif name.endswith(".linear_attention.norm.weight"):
                kind, groups = "GroupedRMSNorm", _integer(text, "linear_num_value_heads")
        if tensor.category == "predictor":
            component = "predictor"
            if ".mhc." not in name:
                position = name.removeprefix("predictor.").removesuffix(".weight")
        if component != "other":
            norm(component, position, kind, dimension, epsilon, groups, tensor.layer)
    if has_gdn and (qwen4 or qwen35):
        for layer in sorted(layer for layer in gdn_layers if layer is not None):
            for position in ("query", "key"):
                norm("gdn", position, "L2Norm", _integer(text, "linear_key_head_dim"), 1e-6, _integer(text, "linear_num_key_heads"), layer)
    for profile in norms.values():
        profile.layers.sort()
    topk_norm = text.get("norm_topk_prob", True if qwen4 else None)
    return AnalysisAdvanced(
        ffn_hidden_size=_integer(text, "intermediate_size") or width("ffn"),
        expert_hidden_size=_integer(text, "moe_intermediate_size") or width("routed_experts"),
        shared_expert_count=shared_count, shared_expert_hidden_size=shared_width,
        hc_count=hc_count if hc_count and hc_count > 1 else None,
        hc_dim=_integer(text, "hidden_size") if hc_count and hc_count > 1 else None,
        hc_lowrank=_integer(text, "hc_lowrank", "hc_rank"), ffn_activation=string("hidden_act"),
        router_activation="softmax" if qwen4 else string("scoring_func", "expert_gating_func") or ("softmax" if qwen35 else None),
        router_normalization="l1_topk" if topk_norm is True else "none" if topk_norm is False else None,
        gdn_conv_activation=(string("hidden_act") or "silu") if has_gdn and (qwen4 or qwen35) else None,
        gdn_conv_type="causal_depthwise_1d" if has_gdn and (qwen4 or qwen35) else None,
        gdn_conv_stride=1 if has_gdn and (qwen4 or qwen35) else None,
        gdn_conv_dilation=1 if has_gdn and (qwen4 or qwen35) else None,
        gdn_gate_activation=(string("output_gate_type", "hidden_act") or "silu") if has_gdn and qwen4 else "silu" if has_gdn and qwen35 else None,
        qsa_gate_activation="sigmoid" if qwen4 and has_qsa else None,
        full_attention_gate_activation="sigmoid" if qwen35 and has_full_gate else None,
        residual_gate_activation="sigmoid" if hc_count and hc_count > 1 and (qwen4 or architecture.startswith("deepseek_v4")) else None,
        indexer_query_heads=_integer(text, "indexer_n_heads", "index_n_heads"),
        indexer_kv_heads=_integer(text, "indexer_kv_heads"), indexer_head_dim=_integer(text, "indexer_head_dim", "index_head_dim"),
        compression_stride=stride,
        compression_strides=[value for value in ratios if isinstance(value, int) and not isinstance(value, bool) and value > 0] if isinstance(ratios, list) else [],
        active_tokens=budget, active_blocks=budget // stride if budget and stride and budget % stride == 0 else _integer(text, "index_topk"),
        linear_key_head_dim=_integer(text, "linear_key_head_dim"), linear_value_head_dim=_integer(text, "linear_value_head_dim"),
        linear_conv_kernel=_integer(text, "linear_conv_kernel_dim"),
        ple_ngram=_integer(text, "ngram_size"), ple_heads=_integer(text, "heads_per_ngram"), ple_conv_kernel=_integer(text, "ple_conv_kernel_size"),
        ple_conv_type="causal_depthwise_1d" if has_ple and qwen4 else None,
        ple_conv_stride=1 if has_ple and qwen4 else None,
        ple_conv_dilation=_integer(text, "ngram_size") if has_ple and qwen4 else None,
        ple_conv_activation="silu" if has_ple and qwen4 else None,
        ple_layer_ids=sorted({tensor.layer for tensor in tensors if tensor.category in {"ple", "ple_projection"} and tensor.layer is not None}),
        predictor_layers=(_integer(text, "mtp_num_hidden_layers") or (_integer(topology, "predictor_layers") if isinstance(topology, dict) else None)) if has_predictor else 0,
        max_context=_integer(text, "max_position_embeddings"), vocab_size=_integer(text, "vocab_size"),
        rope_theta=number(rope, "rope_theta") or number(text, "rope_theta"),
        rope_partial_factor=number(text, "partial_rotary_factor") or number(rope, "partial_rotary_factor"),
        rms_norm_eps=norm_eps, normalizations=list(norms.values()),
    )


def _classify(name: str) -> tuple[str, str, int | None]:
    match = _LAYER.match(name)
    layer = int(match[1]) if match else None
    suffix = name[match.end():] if match else name
    if name.startswith("predictor."):
        category = "predictor"
    elif not name.startswith("model."):
        category = name.split(".", 1)[0]
    elif _is_always_streamed_tensor(name):
        category = "ple"
    elif ".mhc." in name:
        category = "mhc"
    elif ".mlp.experts." in name:
        category = "routed_experts"
    elif ".mlp.shared_expert." in name:
        category = "shared_expert"
    elif ".linear_attention." in name:
        category = "gdn"
    elif ".attention." in name:
        category = "attention"
    elif ".position_embedding." in name or ".associative_memory." in name or ".engram." in name:
        category = "ple_projection"
    elif ".mlp.router." in name:
        category = "router"
    elif ".mlp." in name:
        category = "ffn"
    elif "embedding" in suffix or "embed_tokens" in suffix:
        category = "embedding"
    elif "output.weight" in suffix or "lm_head" in suffix:
        category = "lm_head"
    else:
        category = "other"
    if category == "routed_experts":
        expert = _EXPERT.search(name)
        projection = expert[2] if expert else suffix
    else:
        projection = suffix.removesuffix(".weight").removesuffix(".bias")
        for prefix in ("linear_attention.", "attention.", "mlp.shared_expert.", "mlp.", "model."):
            if projection.startswith(prefix):
                projection = projection[len(prefix):]
                break
        projection = re.sub(r"\.shard\.\d+$", "", projection)
    return category, projection, layer


def _mfq_tensors(artifact: DiscoveredModel) -> tuple[dict[str, Any], dict[str, Any], list[AnalysisTensor], list[AnalysisExpert], list[str]]:
    tensors: list[AnalysisTensor] = []
    experts: list[AnalysisExpert] = []
    warnings: list[str] = []
    with open_mmap(artifact.path) as store:
        def asset(name: str) -> dict[str, Any]:
            record = store.records.get(name)
            if record is None:
                return {}
            if record.nbytes > 4 * 1024 * 1024:
                raise ValueError("analysis JSON asset exceeds 4 MiB")
            value = json.loads(store.read_blob(name))
            return value if isinstance(value, dict) else {}

        config = asset(MODEL_CONFIG_ASSET) or store.header.extra.get("hf_config", {})
        if not isinstance(config, dict):
            config = {}
        graph = asset(MODEL_GRAPH_ASSET)
        for record in store.records.values():
            if is_asset_record(record.name) or record.dtype in {"I32", "I64"}:
                continue
            mapping = map_source_tensor_name(record.name, config)
            name = mapping.canonical_name if mapping else record.name
            category, projection, layer = _classify(name)
            shape: tuple[int, ...] = ()
            family = record.dtype
            parameters = 0
            blob = store.blob_view(record)
            expert_start = len(experts)
            try:
                if record.dtype == "MFE":
                    _magic, count, outputs, _width, _pool_count = struct.unpack_from("<4sIIII", blob)
                    if count > 65_536 or outputs * count > 16_777_216:
                        raise ValueError("MFE analysis geometry is too large")
                    shape, pools = view_mfe_blob(blob)
                    count, outputs, width = shape
                    family = "MFE"
                    try:
                        for pool in pools:
                            metadata = inspect_tensor(pool.dtype, pool.tensor_payload, rows=True)
                            ids = pool.expert_ids.tolist()
                            if math.prod(metadata.shape) != len(ids) * outputs * width:
                                raise ValueError("MFE cohort shape does not match expert bank")
                            if category != "routed_experts" or layer is None:
                                continue
                            storage = metadata.row_bytes
                            if storage is None:
                                storage = np.full(len(ids) * outputs, len(pool.tensor_payload) / (len(ids) * outputs))
                            if storage.size != len(ids) * outputs:
                                raise ValueError("MFE cohort row layout does not match expert bank")
                            overhead = 24 + 4 * len(ids) + len(pool.dtype.encode("ascii")) + len(pool.runtime_payload) + 20 * len(ids) / count
                            storage = storage.reshape(len(ids), outputs) + overhead / (len(ids) * outputs)
                            parts = [(projection, 0, outputs)]
                            if projection == "gate_up":
                                if outputs % 2:
                                    raise ValueError("fused Gate/Up has odd output size")
                                parts = [("gate", 0, outputs // 2), ("up", outputs // 2, outputs)]
                            for index, expert in enumerate(ids):
                                for part, start, end in parts:
                                    size = float(storage[index, start:end].sum())
                                    n = (end - start) * width
                                    experts.append(AnalysisExpert(layer=layer, expert=expert, projection=part,
                                        parameters=n, stored_bytes=size, bpw=8 * size / n,
                                        format=metadata.family, granularity=metadata.granularity))
                    finally:
                        for pool in pools:
                            pool.runtime_payload.release()
                            pool.tensor_payload.release()
                    parameters = math.prod(shape)
                else:
                    metadata = inspect_tensor(record.dtype, blob)
                    shape, family = metadata.shape, metadata.family
                    parameters = math.prod(shape)
                    match = _EXPERT.search(name)
                    if category == "routed_experts" and layer is not None and match:
                        if match[1] is not None:
                            ids, per_expert = [int(match[1])], parameters
                        elif len(shape) == 3:
                            ids, per_expert = list(range(shape[0])), math.prod(shape[1:])
                        else:
                            ids, per_expert = [], 0
                        for expert in ids:
                            parts = [projection] if projection != "gate_up" else ["gate", "up"]
                            for part in parts:
                                size = record.nbytes / len(ids) / len(parts)
                                n = per_expert // len(parts)
                                experts.append(AnalysisExpert(layer=layer, expert=expert, projection=part,
                                    parameters=n, stored_bytes=size, bpw=8 * size / n,
                                    format=family, granularity="tensor"))
            except (ValueError, struct.error, UnicodeError) as error:
                warnings.append(f"{name}: {error}")
                del experts[expert_start:]
                parameters = 0
            finally:
                blob.release()
            tensors.append(AnalysisTensor(name=name, shape=list(shape), category=category, projection=projection,
                layer=layer, parameters=parameters, stored_bytes=record.nbytes,
                bpw=8 * record.nbytes / parameters if parameters else None, format=family))
    return config, graph, tensors, experts, warnings


def _hf_tensors(artifact: DiscoveredModel) -> tuple[dict[str, Any], dict[str, Any], list[AnalysisTensor], list[AnalysisExpert], list[str]]:
    path = artifact.path
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("invalid model config")
    index = path / "model.safetensors.index.json"
    if index.is_file():
        files = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
        paths = [path / name for name in sorted(files)]
    else:
        paths = sorted(path.glob("*.safetensors"))
    tensors, names, warnings = [], [], []
    for shard in paths:
        if not shard.resolve().is_relative_to(path.resolve()):
            raise ValueError("Safetensors shard escapes checkpoint directory")
        with shard.open("rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            if size > 32 * 1024 * 1024:
                raise ValueError("Safetensors header exceeds 32 MiB")
        header = ModelCatalog._safetensors_header(shard)
        for source, item in header.items():
            if source == "__metadata__":
                continue
            names.append(source)
            mapping = map_source_tensor_name(source, config)
            name = mapping.canonical_name if mapping else source
            category, projection, layer = _classify(name)
            shape = item["shape"]
            dtype = item["dtype"]
            if dtype in {"I32", "I64", "U8", "I8", "BOOL"}:
                warnings.append(f"{name}: packed/integer source storage is not a logical parameter shape")
                continue
            parameters = math.prod(shape)
            start, end = item["data_offsets"]
            tensors.append(AnalysisTensor(name=name, shape=shape, category=category, projection=projection,
                layer=layer, parameters=parameters, stored_bytes=end - start,
                bpw=8 * (end - start) / parameters if parameters else None, format=dtype))
    spec = graph_spec_for_source_names(config, names)
    return config, spec.as_dict() if spec else {}, tensors, [], warnings


def analyze_checkpoint(artifact: DiscoveredModel) -> CheckpointAnalysis:
    if not artifact.resource.complete:
        raise ValueError("checkpoint is incomplete; missing shards must be restored before analysis")
    config, graph, tensors, experts, warnings = (_mfq_tensors(artifact) if artifact.resource.format == "mfq" else _hf_tensors(artifact))
    text = config.get("text_config", config)
    if not isinstance(text, dict):
        text = {}
    layer_count = _integer(text, "num_hidden_layers", "n_layer") or 0
    layer_count = max(layer_count, max((item.layer + 1 for item in tensors if item.layer is not None), default=0))
    types = text.get("layer_types", [])
    query_heads, kv_heads = _integer(text, "num_attention_heads"), _integer(text, "num_key_value_heads")
    full_attention = "GQA" if query_heads and kv_heads and query_heads > kv_heads and query_heads % kv_heads == 0 else "MHA" if query_heads and query_heads == kv_heads else "Attention"
    distribution: dict[str, list[int]] = defaultdict(list)
    layers = []
    for layer in range(layer_count):
        weights = [item for item in tensors if item.layer == layer]
        names = [item.name for item in weights]
        if any(".linear_attention." in name for name in names):
            kind = "GDN"
        elif any(".attention.indexer." in name for name in names):
            kind = "QSA"
        elif any(".attention." in name for name in names):
            kind = "MLA" if _integer(text, "q_lora_rank", "kv_lora_rank") else full_attention
        elif layer < len(types):
            kind = str(types[layer])
        else:
            kind = "unknown"
        distribution[kind].append(layer)
        parameters, size = sum(item.parameters for item in weights), sum(item.stored_bytes for item in weights)
        layers.append(AnalysisLayer(layer=layer, attention_type=kind, parameters=parameters, stored_bytes=size,
            average_bpw=8 * size / parameters if parameters and all(item.bpw is not None for item in weights) else None))
    groups: dict[tuple[str, str], list[AnalysisTensor]] = defaultdict(list)
    for item in tensors:
        parts = [item.projection] if item.category != "routed_experts" or item.projection != "gate_up" else ["gate", "up"]
        for part in parts:
            groups[item.category, part].append(item)
    projections = []
    for (category, projection), items in groups.items():
        ids = {item.layer for item in items}
        cells = [cell for cell in experts if cell.projection == projection and cell.layer in ids]
        parameters = sum(cell.parameters for cell in cells) if cells and category == "routed_experts" else sum(item.parameters // (2 if item.projection == "gate_up" else 1) for item in items)
        size = sum(cell.stored_bytes for cell in cells) if cells and category == "routed_experts" else sum(item.stored_bytes / (2 if item.projection == "gate_up" else 1) for item in items)
        projections.append(AnalysisProjection(name=projection, category=category, parameters=parameters, stored_bytes=size,
            average_bpw=8 * size / parameters if parameters and all(item.bpw is not None for item in items) else None,
            tensor_count=len(items), formats=sorted({cell.format for cell in cells}) if cells and category == "routed_experts" else sorted({item.format for item in items})))
    parameters, size = sum(item.parameters for item in tensors), sum(item.stored_bytes for item in tensors)
    if not config:
        warnings.append("checkpoint has no model config; head counts and architecture details are unavailable")
    return CheckpointAnalysis(model_id=artifact.resource.id, name=artifact.resource.name,
        architecture=str(graph.get("architecture") or text.get("model_type") or artifact.resource.architecture),
        format=artifact.resource.format, complete=artifact.resource.complete, layer_count=layer_count,
        hidden_size=_integer(text, "hidden_size"), attention_heads=_integer(text, "num_attention_heads"),
        kv_heads=_integer(text, "num_key_value_heads"), head_dim=_integer(text, "head_dim"),
        linear_key_heads=_integer(text, "linear_num_key_heads"), linear_value_heads=_integer(text, "linear_num_value_heads"),
        expert_count=max(_integer(text, "num_experts", "n_routed_experts") or 0, max((cell.expert + 1 for cell in experts), default=0)),
        experts_per_token=_integer(text, "num_experts_per_tok"), parameters=parameters, stored_bytes=size,
        average_bpw=8 * size / parameters if parameters and all(item.bpw is not None for item in tensors) else None,
        attention_distribution=dict(distribution), projections=sorted(projections, key=lambda item: (item.category, item.name)),
        layers=layers, experts=experts, tensors=tensors,
        advanced=_advanced(text, graph, tensors),
        cache_profile=cache_profile(config, include_recurrent=False),
        graph={**graph, "hc_count": _integer(text, "hc_count", "hc_mult"),
               "has_ple": any(item.category == "ple" for item in tensors),
               "has_shared_expert": any(item.category == "shared_expert" for item in tensors)}, warnings=warnings)
