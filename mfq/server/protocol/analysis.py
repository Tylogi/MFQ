from __future__ import annotations

from typing import Any

from pydantic import Field

from mfq.server.protocol.models import ModelCacheProfile, ProtocolModel


class AnalysisProjection(ProtocolModel):
    name: str
    category: str
    parameters: int
    stored_bytes: float
    average_bpw: float | None
    tensor_count: int
    formats: list[str]


class AnalysisLayer(ProtocolModel):
    layer: int
    attention_type: str
    parameters: int
    stored_bytes: float
    average_bpw: float | None


class AnalysisExpert(ProtocolModel):
    layer: int
    expert: int
    projection: str
    parameters: int
    stored_bytes: float
    bpw: float
    format: str
    granularity: str


class AnalysisTensor(ProtocolModel):
    name: str
    shape: list[int]
    category: str
    projection: str
    layer: int | None
    parameters: int
    stored_bytes: int
    bpw: float | None
    format: str


class AnalysisNormalization(ProtocolModel):
    component: str
    position: str
    kind: str | None = None
    dimension: int | None = None
    epsilon: float | None = None
    groups: int | None = None
    layers: list[int] = Field(default_factory=list)


class AnalysisAdvanced(ProtocolModel):
    ffn_hidden_size: int | None = None
    expert_hidden_size: int | None = None
    shared_expert_count: int | None = None
    shared_expert_hidden_size: int | None = None
    hc_count: int | None = None
    hc_dim: int | None = None
    hc_lowrank: int | None = None
    ffn_activation: str | None = None
    router_activation: str | None = None
    router_normalization: str | None = None
    gdn_conv_activation: str | None = None
    gdn_conv_type: str | None = None
    gdn_conv_stride: int | None = None
    gdn_conv_dilation: int | None = None
    gdn_gate_activation: str | None = None
    qsa_gate_activation: str | None = None
    full_attention_gate_activation: str | None = None
    residual_gate_activation: str | None = None
    indexer_query_heads: int | None = None
    indexer_kv_heads: int | None = None
    indexer_head_dim: int | None = None
    compression_stride: int | None = None
    compression_strides: list[int] = Field(default_factory=list)
    active_tokens: int | None = None
    active_blocks: int | None = None
    linear_key_head_dim: int | None = None
    linear_value_head_dim: int | None = None
    linear_conv_kernel: int | None = None
    ple_ngram: int | None = None
    ple_heads: int | None = None
    ple_conv_kernel: int | None = None
    ple_conv_type: str | None = None
    ple_conv_stride: int | None = None
    ple_conv_dilation: int | None = None
    ple_conv_activation: str | None = None
    ple_layer_ids: list[int] = Field(default_factory=list)
    predictor_layers: int | None = None
    max_context: int | None = None
    vocab_size: int | None = None
    rope_theta: float | None = None
    rope_partial_factor: float | None = None
    rms_norm_eps: float | None = None
    normalizations: list[AnalysisNormalization] = Field(default_factory=list)


class CheckpointAnalysis(ProtocolModel):
    model_id: str
    name: str
    architecture: str
    format: str
    complete: bool
    layer_count: int
    hidden_size: int | None
    attention_heads: int | None
    kv_heads: int | None
    head_dim: int | None
    linear_key_heads: int | None
    linear_value_heads: int | None
    expert_count: int
    experts_per_token: int | None
    parameters: int
    stored_bytes: int
    average_bpw: float | None
    attention_distribution: dict[str, list[int]]
    projections: list[AnalysisProjection]
    layers: list[AnalysisLayer]
    experts: list[AnalysisExpert]
    tensors: list[AnalysisTensor]
    graph: dict[str, Any]
    advanced: AnalysisAdvanced = Field(default_factory=AnalysisAdvanced)
    cache_profile: ModelCacheProfile | None = None
    warnings: list[str]
