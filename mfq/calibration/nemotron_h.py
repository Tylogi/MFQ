"""Nemotron-H linear targets for activation-imatrix collection.

Linear targets keep the HF source tensor name, which is what the HF import
boundary also carries as ``source_name``.  Routed experts are one 3-D bank in
the MFQ container, so they use the canonical bank name and record one
importance row per expert.
"""

from __future__ import annotations

from mfq.architectures.nemotron_h import NemotronHConfig
from mfq.calibration.imatrix import ImatrixTarget
from mfq.calibration.qwen35 import HfSafetensorIndex


def _source(index: HfSafetensorIndex, name: str) -> tuple[int, ...]:
    if name not in index.weight_map:
        raise KeyError(f"Nemotron-H checkpoint has no tensor {name!r}")
    return index.shape(name)


def _add(
    targets: list[ImatrixTarget],
    index: HfSafetensorIndex,
    prefix: str,
    suffix: str,
    module_name: str,
    *,
    kind: str = "linear",
    name: str | None = None,
) -> None:
    source = prefix + suffix
    shape = _source(index, source)
    targets.append(
        ImatrixTarget(
            source if name is None else name,
            module_name,
            int(shape[-1]),
            kind=kind,
        )
    )


def nemotron_h_imatrix_targets(
    index: HfSafetensorIndex,
    contract: NemotronHConfig,
) -> tuple[dict[int, tuple[ImatrixTarget, ...]], tuple[ImatrixTarget, ...]]:
    """Return per-layer imatrix targets for the text backbone only."""

    by_layer: dict[int, list[ImatrixTarget]] = {}
    for layer, mixer in enumerate(contract.mixer_types):
        prefix = f"language_model.backbone.layers.{layer}."
        values: list[ImatrixTarget] = []

        if mixer == "linear_attention":
            _add(values, index, prefix, "mixer.in_proj.weight", "mixer.in_proj")
            _add(values, index, prefix, "mixer.out_proj.weight", "mixer.out_proj")
        elif mixer == "full_attention":
            for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
                _add(
                    values,
                    index,
                    prefix,
                    f"mixer.{projection}.weight",
                    f"mixer.{projection}",
                )
        elif mixer == "mlp":
            _add(values, index, prefix, "mixer.up_proj.weight", "mixer.up_proj")
            _add(values, index, prefix, "mixer.down_proj.weight", "mixer.down_proj")
        elif mixer == "moe":
            _add(
                values,
                index,
                prefix,
                "mixer.gate.weight",
                "mixer.gate",
                kind="router",
            )
            _add(
                values,
                index,
                prefix,
                "mixer.fc1_latent_proj.weight",
                "mixer.fc1_latent_proj",
            )
            _add(
                values,
                index,
                prefix,
                "mixer.fc2_latent_proj.weight",
                "mixer.fc2_latent_proj",
            )
            if contract.n_shared_experts > 0:
                _add(
                    values,
                    index,
                    prefix,
                    "mixer.shared_experts.up_proj.weight",
                    "mixer.shared_experts.up_proj",
                )
                _add(
                    values,
                    index,
                    prefix,
                    "mixer.shared_experts.down_proj.weight",
                    "mixer.shared_experts.down_proj",
                )
            for projection, dimension in (
                ("up", contract.moe_latent_size),
                ("down", contract.moe_intermediate_size),
            ):
                source_names = [
                    prefix + f"mixer.experts.{expert}.{projection}_proj.weight"
                    for expert in range(contract.num_experts)
                ]
                for source_name in source_names:
                    _source(index, source_name)
                values.append(
                    ImatrixTarget(
                        f"model.block.{layer}.mlp.experts.{projection}.weight",
                        "mixer.experts",
                        dimension,
                        contract.num_experts,
                        f"expert_{projection}",
                    )
                )
        else:
            raise ValueError(f"unsupported Nemotron-H mixer type: {mixer!r}")
        by_layer[layer] = values

    all_targets = tuple(
        target for layer in range(contract.num_hidden_layers) for target in by_layer[layer]
    )
    return {key: tuple(value) for key, value in by_layer.items()}, all_targets


__all__ = ["nemotron_h_imatrix_targets"]
