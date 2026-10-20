"""Nemotron-H backend for disk-backed layerwise calibration replay."""

from __future__ import annotations

import gc
import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn

from mfq.architectures.hf_config import load_hf_model_config
from mfq.architectures.nemotron_h import NemotronHConfig
from mfq.calibration.masking import create_causal_mask_compat
from mfq.calibration.nemotron_h import nemotron_h_imatrix_targets
from mfq.calibration.qwen35 import HfSafetensorIndex
from mfq.calibration.ssm import selective_scan_chunked

_EXPERT_BANK = re.compile(r"^mixer\.experts\.(\d+)\.(up|down)_proj\.weight$")
_EMBEDDING = "language_model.backbone.embeddings.weight"


def _transformers_config(contract: NemotronHConfig) -> object:
    """Build the Transformers config from the validated MFQ contract."""

    from transformers import NemotronHConfig as HfNemotronHConfig

    return HfNemotronHConfig(
        vocab_size=contract.vocab_size,
        hidden_size=contract.hidden_size,
        layers_block_type=list(contract.mixer_types),
        tie_word_embeddings=contract.tie_word_embeddings,
        num_attention_heads=contract.num_attention_heads,
        num_key_value_heads=contract.num_key_value_heads,
        head_dim=contract.head_dim,
        max_position_embeddings=contract.max_position_embeddings,
        attention_bias=contract.attention_bias,
        intermediate_size=contract.intermediate_size,
        mlp_hidden_act=contract.expert_activation,
        mlp_bias=False,
        use_mamba_kernels=False,
        ssm_state_size=contract.ssm_state_size,
        mamba_num_heads=contract.mamba_num_heads,
        mamba_head_dim=contract.mamba_head_dim,
        mamba_hidden_act="silu",
        n_groups=contract.mamba_groups,
        conv_kernel=contract.conv_kernel,
        time_step_min=contract.time_step_init_limit[0],
        time_step_max=contract.time_step_init_limit[1],
        time_step_limit=list(contract.time_step_limit),
        time_step_floor=contract.time_step_floor,
        use_conv_bias=contract.conv_bias,
        chunk_size=contract.chunk_size,
        mamba_proj_bias=False,
        mamba_ssm_cache_dtype="float32",
        n_routed_experts=max(contract.num_experts, 1),
        n_shared_experts=contract.n_shared_experts,
        moe_intermediate_size=max(contract.moe_intermediate_size, 1),
        moe_shared_expert_intermediate_size=(
            contract.moe_shared_intermediate_size
            if contract.n_shared_experts > 0
            else 1
        ),
        moe_latent_size=contract.moe_latent_size or None,
        num_experts_per_tok=max(contract.num_experts_per_tok, 1),
        routed_scaling_factor=contract.routed_scaling_factor,
        n_group=contract.expert_group_count,
        topk_group=contract.expert_group_topk,
        norm_topk_prob=contract.norm_topk_prob,
        num_nextn_predict_layers=0,
        use_bias=False,
        layer_norm_epsilon=contract.norm_epsilon,
        residual_in_fp32=contract.residual_in_fp32,
    )


class NemotronHLayerwiseBackend:
    """Load one original Nemotron-H text decoder layer at a time."""

    teacher_dtype = torch.bfloat16
    embedding_name = _EMBEDDING

    def __init__(
        self,
        model_path: str | Path,
        scheme: object | None = None,
        *,
        device: str | torch.device = "cuda:0",
        attention: str = "sdpa",
        teacher_dtype: torch.dtype | None = None,
        scan_chunk_size: int | None = None,
    ) -> None:
        if scheme is not None:
            raise NotImplementedError("Nemotron-H layerwise replay supports BF16 calibration only")
        self.root = Path(model_path).resolve()
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Nemotron-H layerwise replay requested CUDA without CUDA support")
        if attention not in {"eager", "sdpa"}:
            raise ValueError("attention must be eager or sdpa")

        raw = load_hf_model_config(self.root)
        self.contract = NemotronHConfig.from_hf_config(raw)
        self.config = _transformers_config(self.contract)
        self.config._attn_implementation = attention
        if teacher_dtype is not None:
            self.teacher_dtype = teacher_dtype
        self.num_layers = self.contract.num_hidden_layers
        self.hidden_size = self.contract.hidden_size
        if scan_chunk_size is None:
            scan_chunk_size = min(int(self.contract.chunk_size), 256)
        if scan_chunk_size <= 0:
            raise ValueError("Nemotron-H scan chunk size must be positive")
        self.scan_chunk_size = int(scan_chunk_size)
        self.index = HfSafetensorIndex(self.root)
        self._embedding: torch.Tensor | None = None

    def initial_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self._embedding is None:
            self._embedding = self.index.tensor(_EMBEDDING, device="cpu")
        ids = input_ids.detach().to(device="cpu", dtype=torch.int64)
        hidden = self._embedding.index_select(0, ids.reshape(-1)).reshape(
            *ids.shape,
            self.hidden_size,
        )
        return hidden.to(device=self.device, dtype=self.teacher_dtype)

    def release_initial_state(self) -> None:
        self._embedding = None
        gc.collect()

    def imatrix_targets(self):
        """Return the per-layer activation-imatrix targets for this model."""

        return nemotron_h_imatrix_targets(self.index, self.contract)

    def _layer_state(self, layer_index: int) -> dict[str, torch.Tensor]:
        prefix = f"language_model.backbone.layers.{layer_index}."
        banks: dict[str, dict[int, str]] = {"up_proj": {}, "down_proj": {}}
        direct: list[str] = []
        for name in self.index.weight_map:
            if not name.startswith(prefix):
                continue
            suffix = name[len(prefix) :]
            match = _EXPERT_BANK.match(suffix)
            if match is None:
                direct.append(name)
                continue
            banks[match.group(2) + "_proj"][int(match.group(1))] = name

        state: dict[str, torch.Tensor] = {
            name[len(prefix) :]: self.index.tensor(name) for name in direct
        }
        for projection, expert_names in sorted(banks.items()):
            if not expert_names:
                continue
            expected = list(range(len(expert_names)))
            if sorted(expert_names) != expected:
                raise ValueError(
                    f"Nemotron-H layer {layer_index} {projection} expert set is not "
                    f"contiguous: {sorted(expert_names)[:8]}"
                )
            first = self.index.tensor(expert_names[0])
            shape = tuple(int(value) for value in first.shape)
            if any(self.index.shape(expert_names[expert]) != shape for expert in expected):
                raise ValueError(
                    f"Nemotron-H layer {layer_index} {projection} experts disagree on shape"
                )
            bank = torch.empty((len(expected), *shape), dtype=first.dtype)
            bank[0].copy_(first)
            del first
            for expert in expected[1:]:
                bank[expert].copy_(self.index.tensor(expert_names[expert]))
            state[f"mixer.experts.{projection}"] = bank
        if (
            self.contract.mixer_types[layer_index] == "moe"
            and self.contract.n_shared_experts == 0
        ):
            shared = int(self.config.moe_shared_expert_intermediate_size)
            state["mixer.shared_experts.up_proj.weight"] = torch.zeros(
                (shared, self.hidden_size),
            )
            state["mixer.shared_experts.down_proj.weight"] = torch.zeros(
                (self.hidden_size, shared),
            )
        return state

    def _load_dense_layer(self, layer_index: int) -> nn.Module:
        from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHBlock

        with torch.device("meta"):
            layer = NemotronHBlock(self.config, layer_index)
        state = self._layer_state(layer_index)
        result = layer.load_state_dict(state, strict=True, assign=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(
                f"Nemotron-H layer {layer_index} state mismatch: "
                f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
            )
        del state
        return layer.to(device=self.device, dtype=self.teacher_dtype).eval().requires_grad_(False)

    @contextmanager
    def layer(self, layer_index: int, *, quantized: bool) -> Iterator[nn.Module]:
        if quantized:
            raise NotImplementedError("Nemotron-H streamed calibration supports BF16 layers only")
        if layer_index < 0 or layer_index >= self.num_layers:
            raise IndexError(f"Nemotron-H layer {layer_index} is out of range")
        layer = self._load_dense_layer(layer_index)
        try:
            yield layer
        finally:
            del layer
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
                torch.cuda.empty_cache()

    def _mamba_forward(
        self,
        mixer: nn.Module,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, sequence, _hidden = hidden_states.shape
        inner = int(mixer.intermediate_size)
        conv_dim = int(mixer.conv_dim)
        heads = int(mixer.num_heads)
        groups = int(mixer.n_groups)
        head_dim = int(mixer.head_dim)
        state_size = int(mixer.ssm_state_size)
        kernel = int(mixer.conv_kernel_size)
        if groups <= 0 or heads % groups:
            raise ValueError(f"Nemotron-H Mamba heads {heads} do not divide into {groups} groups")
        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask[:, :, None].to(hidden_states.dtype)
        rate = -torch.exp(mixer.A_log.float())
        output = torch.empty(
            (batch, sequence, self.hidden_size),
            device=hidden_states.device,
            dtype=mixer.out_proj.weight.dtype,
        )
        history: torch.Tensor | None = None
        scan_state: torch.Tensor | None = None
        for start in range(0, sequence, self.scan_chunk_size):
            end = min(start + self.scan_chunk_size, sequence)
            length = end - start
            projected = mixer.in_proj(hidden_states[:, start:end])
            gate, projected_bc, time_step = torch.split(
                projected,
                (inner, conv_dim, heads),
                dim=-1,
            )
            conv_input = (
                projected_bc
                if history is None
                else torch.cat((history, projected_bc), dim=1)
            )
            offset = 0 if history is None else kernel - 1
            convolved = mixer.conv1d(conv_input.transpose(1, 2))
            convolved = convolved[..., offset : offset + length].transpose(1, 2)
            if kernel > 1:
                history = conv_input[:, -(kernel - 1) :]
            values, recurrent_b, recurrent_c = torch.split(
                mixer.act(convolved),
                (inner, groups * state_size, groups * state_size),
                dim=-1,
            )
            if attention_mask is not None:
                values = values * attention_mask[:, start:end, None].to(values.dtype)
            recurrent_b = recurrent_b.reshape(batch, length, groups, state_size)
            recurrent_c = recurrent_c.reshape(batch, length, groups, state_size)
            recurrent_b = recurrent_b.repeat_interleave(heads // groups, dim=2)
            recurrent_c = recurrent_c.repeat_interleave(heads // groups, dim=2)
            time_step = torch.nn.functional.softplus(time_step + mixer.dt_bias)
            time_step = torch.clamp(
                time_step,
                min=float(mixer.time_step_limit[0]),
                max=float(mixer.time_step_limit[1]),
            )
            scanned, scan_state = selective_scan_chunked(
                values.reshape(batch, length, heads, head_dim),
                time_step,
                rate,
                recurrent_b,
                recurrent_c,
                skip=mixer.D,
                chunk_size=self.scan_chunk_size,
                initial_state=scan_state,
                return_state=True,
            )
            # Zamba2RMSNormGated multiplies silu(gate) before the grouped
            # RMS: the checkpoint invariant is norm_before_gate=False.
            scanned = mixer.norm(scanned.reshape(batch, length, inner), gate)
            output[:, start:end] = mixer.out_proj(
                scanned.to(mixer.out_proj.weight.dtype)
            )
        return output

    def _forward_layer(
        self,
        layer: nn.Module,
        layer_index: int,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from transformers.masking_utils import (
            create_causal_mask,
            create_recurrent_attention_mask,
        )

        batch, sequence, _hidden = hidden_states.shape
        position_ids = (
            torch.arange(sequence, device=self.device, dtype=torch.int64)
            .unsqueeze(0)
            .expand(batch, -1)
        )
        residual = hidden_states
        normalized = layer.norm(hidden_states.to(dtype=layer.norm.weight.dtype))
        mixer_type = self.contract.mixer_types[layer_index]
        if mixer_type == "linear_attention":
            recurrent_mask = create_recurrent_attention_mask(
                config=self.config,
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
                past_key_values=None,
            )
            mixer_output = self._mamba_forward(layer.mixer, normalized, recurrent_mask)
        elif mixer_type == "full_attention":
            causal_mask = create_causal_mask_compat(
                create_causal_mask,
                config=self.config,
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
                past_key_values=None,
                position_ids=position_ids,
            )
            mixer_output, _weights = layer.mixer(
                hidden_states=normalized,
                attention_mask=causal_mask,
                position_ids=position_ids,
                use_cache=False,
            )
        elif mixer_type in {"moe", "mlp"}:
            mixer_output = layer.mixer(normalized)
        else:
            raise ValueError(f"unsupported Nemotron-H mixer type: {mixer_type!r}")
        return residual + mixer_output

    @torch.inference_mode()
    def forward_layer(
        self,
        layer: nn.Module,
        layer_index: int,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._forward_layer(
            layer,
            layer_index,
            hidden_states,
            attention_mask=attention_mask,
        )


__all__ = ["NemotronHLayerwiseBackend"]
