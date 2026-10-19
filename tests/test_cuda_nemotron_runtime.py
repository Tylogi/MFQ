"""Nemotron-H text-backbone oracle tests (NumPy reference + native bridge).

The reference implements the released text graph with the runtime's dtype
boundaries: dense float32 projections, float16 attention operands and cache,
and the float16 activation boundary of the shared MoE FFN. The native side is
``mfq-diagnostics --check-nemotron-h`` reading a synthetic ``.mfq`` fixture.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

HIDDEN = 64
MAMBA_HEADS = 4
MAMBA_HEAD_DIM = 8
MAMBA_GROUPS = 2
SSM_STATE = 32
CONV_KERNEL = 4
ATTENTION_HEADS = 4
KV_HEADS = 2
HEAD_DIM = 16
EXPERTS = 3
TOP_K = 2
MOE_INTERMEDIATE = 8
MOE_LATENT = 16
SHARED_INTERMEDIATE = 12
DENSE_INTERMEDIATE = 32
VOCAB = 32

LAYER_MIXERS = ("mamba", "moe", "attention", "moe", "mamba", "attention")
MTP_MIXERS = ("attention", "moe")
MTP_COMPONENT = dict(kind="predictor", tensor_root="predictor",
                     implementation="nemotron_h_mtp", policy="optional")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def half(x):
    """Round a float32 array through the runtime's float16 operand boundary."""

    return np.asarray(x, np.float32).astype(np.float16).astype(np.float32)


def rms_norm(x, weight, eps):
    x = np.asarray(x, np.float32)
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * weight


def config_dict(mixers, mtp=False):
    llm = dict(
        model_type="nemotron_h",
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=DENSE_INTERMEDIATE,
        num_attention_heads=ATTENTION_HEADS,
        num_key_value_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        attention_bias=False,
        max_position_embeddings=32,
        layer_norm_epsilon=1e-5,
        layers_block_type=list(mixers),
        mamba_num_heads=MAMBA_HEADS,
        mamba_head_dim=MAMBA_HEAD_DIM,
        ssm_state_size=SSM_STATE,
        n_groups=MAMBA_GROUPS,
        conv_kernel=CONV_KERNEL,
        use_conv_bias=True,
        mamba_hidden_act="silu",
        mamba_proj_bias=False,
        mamba_ssm_cache_dtype="float32",
        chunk_size=4,
        time_step_min=1e-3,
        time_step_max=1e-1,
        time_step_floor=1e-4,
        time_step_limit=[0.0, {"__float__": "Infinity"}],
        n_routed_experts=EXPERTS,
        num_experts_per_tok=TOP_K,
        n_group=1,
        topk_group=1,
        moe_intermediate_size=MOE_INTERMEDIATE,
        moe_shared_expert_intermediate_size=SHARED_INTERMEDIATE,
        moe_latent_size=MOE_LATENT,
        moe_shared_expert_overlap=False,
        mlp_hidden_act="relu2",
        norm_topk_prob=True,
        routed_scaling_factor=5.0,
        n_shared_experts=1,
        use_bias=False,
        mlp_bias=False,
        num_nextn_predict_layers=0,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        tie_word_embeddings=False,
        eos_token_id=11,
    )
    if mtp:
        # One Megatron-Core MultiTokenPredictionLayer with two sub-blocks.
        llm["num_nextn_predict_layers"] = 1
        llm["mtp_layers_block_type"] = list(MTP_MIXERS)
        # Speculation needs a prompt plus a drafted window of context.
        llm["max_position_embeddings"] = 64
    return dict(
        model_type="nemotron_h_omni",
        llm_config=llm,
        tie_word_embeddings=False,
        eos_token_id=11,
    )


def mixer_weights(rng, prefix, kind):
    """Build one pre-normed mixer block under a canonical block prefix."""

    def rand(shape, scale=0.08):
        return rng.normal(scale=scale, size=shape).astype(np.float32)

    weights = {prefix + ".attention.norm.weight": np.ones(HIDDEN, np.float32)}
    inner = MAMBA_HEADS * MAMBA_HEAD_DIM
    conv_width = inner + 2 * MAMBA_GROUPS * SSM_STATE
    projection = inner + conv_width + MAMBA_HEADS
    if kind == "mamba":
        name = prefix + ".linear_attention."
        weights[name + "in_projection.weight"] = rand((projection, HIDDEN))
        weights[name + "conv.weight"] = rand((conv_width, 1, CONV_KERNEL), 0.05)
        weights[name + "conv.bias"] = rand((conv_width,), 0.02)
        weights[name + "a"] = rand((MAMBA_HEADS,), 0.05)
        weights[name + "d"] = rand((MAMBA_HEADS,), 0.05)
        weights[name + "dt_bias"] = rand((MAMBA_HEADS,), 0.05)
        weights[name + "norm.weight"] = np.ones(inner, np.float32)
        weights[name + "output.weight"] = rand((HIDDEN, inner))
    elif kind == "attention":
        name = prefix + ".attention."
        weights[name + "query.weight"] = rand((HIDDEN, HIDDEN))
        weights[name + "key.weight"] = rand((KV_HEADS * HEAD_DIM, HIDDEN))
        weights[name + "value.weight"] = rand((KV_HEADS * HEAD_DIM, HIDDEN))
        weights[name + "output.weight"] = rand((HIDDEN, HIDDEN))
    elif kind == "moe":
        name = prefix + ".mlp."
        weights[name + "router.weight"] = rand((EXPERTS, HIDDEN))
        weights[name + "router.bias"] = rand((EXPERTS,), 0.02)
        weights[name + "latent.down.weight"] = rand((MOE_LATENT, HIDDEN))
        weights[name + "latent.up.weight"] = rand((HIDDEN, MOE_LATENT))
        weights[name + "experts.up.weight"] = rand((EXPERTS, MOE_INTERMEDIATE, MOE_LATENT))
        weights[name + "experts.down.weight"] = rand((EXPERTS, MOE_LATENT, MOE_INTERMEDIATE))
        weights[name + "shared_expert.up.weight"] = rand((SHARED_INTERMEDIATE, HIDDEN))
        weights[name + "shared_expert.down.weight"] = rand((HIDDEN, SHARED_INTERMEDIATE))
    elif kind == "mlp":
        name = prefix + ".mlp."
        weights[name + "up.weight"] = rand((DENSE_INTERMEDIATE, HIDDEN))
        weights[name + "down.weight"] = rand((HIDDEN, DENSE_INTERMEDIATE))
    else:
        raise AssertionError(kind)
    return weights


def model_fixture(mixers=LAYER_MIXERS, mtp=False):
    from mfq.architectures.nemotron_h import NemotronHConfig

    config = config_dict(mixers, mtp)
    # The fixture must satisfy the same contract as a converted checkpoint.
    NemotronHConfig.from_hf_config(config)
    rng = np.random.default_rng(5306)

    def rand(shape, scale=0.08):
        return rng.normal(scale=scale, size=shape).astype(np.float32)

    weights = {
        "model.token_embedding.weight": rand((VOCAB, HIDDEN), 0.2),
        "model.output.weight": rand((VOCAB, HIDDEN), 0.1),
        "model.output_norm.weight": np.ones(HIDDEN, np.float32),
    }
    for layer, kind in enumerate(mixers):
        weights.update(mixer_weights(rng, f"model.block.{layer}", kind))
    if mtp:
        # output_norm must not be uniform: the references hand the MTP head the
        # post-output_norm hidden, and with a uniform norm weight that value is
        # only a rescaling of the pre-norm hidden (RMSNorm is scale invariant),
        # which would hide a wrong hidden source from the oracle.
        weights["model.output_norm.weight"] = (
            np.linspace(0.5, 1.5, HIDDEN).astype(np.float32))
        weights["predictor.embedding_norm.weight"] = np.ones(HIDDEN, np.float32)
        weights["predictor.hidden_norm.weight"] = np.ones(HIDDEN, np.float32)
        # Non-uniform as well: the depth chain must feed the head's finalized
        # output, which only differs from its pre-norm input beyond a rescaling
        # when the final norm weight is not constant.
        weights["predictor.output_norm.weight"] = (
            np.linspace(0.6, 1.4, HIDDEN).astype(np.float32))
        # eh_proj reads cat([embedding, hidden]) and keeps the hidden width.
        weights["predictor.fusion.weight"] = rand((HIDDEN, 2 * HIDDEN))
        for layer, kind in enumerate(MTP_MIXERS):
            weights.update(mixer_weights(rng, f"predictor.block.{layer}", kind))
    return config, weights


def mamba_reference(x, weights, prefix, contract, eps):
    batch, tokens, _ = x.shape
    inner = MAMBA_HEADS * MAMBA_HEAD_DIM
    state_width = SSM_STATE
    groups = MAMBA_GROUPS
    conv_width = inner + 2 * groups * state_width
    projection = x @ weights[prefix + "in_projection.weight"].T
    gate = projection[..., :inner]
    conv_input = projection[..., inner:inner + conv_width]
    dt = projection[..., inner + conv_width:]
    # The convolution starts from an empty state and is causal: tap j of the
    # [C,1,K] kernel weights the input lag K-1-j.
    convolved = np.zeros((batch, tokens, conv_width), np.float32)
    for tap in range(CONV_KERNEL):
        lag = CONV_KERNEL - 1 - tap
        shifted = np.zeros_like(convolved)
        if lag < tokens:
            shifted[:, lag:] = conv_input[:, :tokens - lag]
        convolved += shifted * weights[prefix + "conv.weight"][:, 0, tap]
    convolved = convolved + weights[prefix + "conv.bias"]
    convolved = convolved * sigmoid(convolved)
    scan_input = convolved[..., :inner].reshape(batch, tokens, MAMBA_HEADS, MAMBA_HEAD_DIM)
    b = convolved[..., inner:inner + groups * state_width].reshape(
        batch, tokens, groups, state_width)
    c = convolved[..., inner + groups * state_width:].reshape(
        batch, tokens, groups, state_width)
    repeat = MAMBA_HEADS // groups
    b = np.repeat(b, repeat, axis=2)
    c = np.repeat(c, repeat, axis=2)
    a = -np.exp(weights[prefix + "a"].astype(np.float32))
    lo, hi = contract.time_step_limit
    step = np.clip(softplus(dt + weights[prefix + "dt_bias"]), lo, hi)
    skip = weights[prefix + "d"].astype(np.float32).reshape(1, 1, MAMBA_HEADS, 1)
    state = np.zeros((batch, MAMBA_HEADS, MAMBA_HEAD_DIM, state_width), np.float32)
    scanned = np.zeros((batch, tokens, MAMBA_HEADS, MAMBA_HEAD_DIM), np.float32)
    for token in range(tokens):
        decay = np.exp(step[:, token, :, None, None] * a[None, :, None, None])
        state = state * decay + (step[:, token, :, None, None] * b[:, token, :, None, :]) * \
            scan_input[:, token, :, :, None]
        scanned[:, token] = np.sum(state * c[:, token, :, None, :], axis=-1) + \
            skip[:, 0] * scan_input[:, token]
    gated = scanned.reshape(batch * tokens, inner) * \
        (gate * sigmoid(gate)).reshape(batch * tokens, inner)
    grouped = gated.reshape(batch * tokens, groups, inner // groups)
    normalized = grouped / np.sqrt(np.mean(grouped * grouped, axis=-1, keepdims=True) + eps)
    normalized = normalized.reshape(batch * tokens, inner) * weights[prefix + "norm.weight"]
    return (normalized @ weights[prefix + "output.weight"].T).reshape(x.shape)


def attention_reference(x, weights, prefix, cache, offset):
    batch, tokens, _ = x.shape
    q = half(x @ weights[prefix + "query.weight"].T).reshape(
        batch, tokens, ATTENTION_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    k = half(x @ weights[prefix + "key.weight"].T).reshape(
        batch, tokens, KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    v = half(x @ weights[prefix + "value.weight"].T).reshape(
        batch, tokens, KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    cache[0][:, :, offset:offset + tokens] = k
    cache[1][:, :, offset:offset + tokens] = v
    visible = offset + tokens
    keys = cache[0][:, :, :visible]
    values = cache[1][:, :, :visible]
    repeat = ATTENTION_HEADS // KV_HEADS
    keys = np.repeat(keys, repeat, axis=1)
    values = np.repeat(values, repeat, axis=1)
    scores = np.einsum("bhqd,bhkd->bhqk", q, keys) / math.sqrt(HEAD_DIM)
    queries = np.arange(offset, offset + tokens)
    allowed = np.arange(visible)[None, :] <= queries[:, None]
    scores = np.where(allowed[None, None], scores, -np.inf)
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    probabilities = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
    attended = half(np.einsum("bhqk,bhkd->bhqd", probabilities, values))
    attended = attended.transpose(0, 2, 1, 3).reshape(batch, tokens, -1)
    return attended @ weights[prefix + "output.weight"].T


def moe_reference(x, weights, prefix, contract):
    # The shared FFN casts its activation to float16 once at its entry.
    flat = half(x).reshape(-1, HIDDEN)
    logits = flat @ weights[prefix + "router.weight"].T
    scores = sigmoid(logits)
    choice = scores + weights[prefix + "router.bias"]
    ids = np.argsort(-choice, axis=-1, kind="stable")[:, :TOP_K]
    selected = np.take_along_axis(scores, ids, axis=-1)
    selected = selected / np.maximum(selected.sum(axis=-1, keepdims=True), 1e-20)
    selected = selected * contract.routed_scaling_factor
    latent = flat @ weights[prefix + "latent.down.weight"].T
    up = np.einsum("troi,ti->tro", weights[prefix + "experts.up.weight"][ids], latent)
    hidden = np.maximum(up, 0.0) ** 2
    down = np.einsum("troi,tri->tro", weights[prefix + "experts.down.weight"][ids], hidden)
    routed = (down * selected[..., None]).sum(axis=1)
    routed = routed @ weights[prefix + "latent.up.weight"].T
    shared = np.maximum(flat @ weights[prefix + "shared_expert.up.weight"].T, 0.0) ** 2
    shared = half(shared @ weights[prefix + "shared_expert.down.weight"].T)
    return (routed + shared).reshape(x.shape)


def dense_mlp_reference(x, weights, prefix):
    flat = half(x).reshape(-1, HIDDEN)
    hidden = np.maximum(flat @ weights[prefix + "up.weight"].T, 0.0) ** 2
    return (hidden @ weights[prefix + "down.weight"].T).reshape(x.shape)


def model_reference(config, weights):
    from mfq.architectures.nemotron_h import NemotronHConfig

    contract = NemotronHConfig.from_hf_config(config)
    ids = np.arange(1, 8)[None]
    hidden = weights["model.token_embedding.weight"][ids]
    cache = [np.zeros((1, KV_HEADS, 32, HEAD_DIM), np.float16),
             np.zeros((1, KV_HEADS, 32, HEAD_DIM), np.float16)]
    offset = 0
    for layer, kind in enumerate(contract.mixer_types):
        prefix = f"model.block.{layer}."
        normalized = rms_norm(hidden, weights[prefix + "attention.norm.weight"],
                              contract.norm_epsilon)
        if kind == "linear_attention":
            branch = mamba_reference(normalized, weights, prefix + "linear_attention.",
                                     contract, contract.norm_epsilon)
        elif kind == "full_attention":
            branch = attention_reference(normalized, weights, prefix + "attention.", cache,
                                         offset)
        elif kind == "moe":
            branch = moe_reference(normalized, weights, prefix + "mlp.", contract)
        else:
            branch = dense_mlp_reference(normalized, weights, prefix + "mlp.")
        hidden = hidden + branch
    finalized = rms_norm(hidden, weights["model.output_norm.weight"], contract.norm_epsilon)
    logits = finalized @ weights["model.output.weight"].T
    return finalized, logits


def mtp_head(config, weights, hidden, ids, cache, offset):
    """One MTP head step against an explicit predictor KV history."""

    from mfq.architectures.nemotron_h import NemotronHConfig

    contract = NemotronHConfig.from_hf_config(config)
    eps = contract.norm_epsilon
    embedded = weights["model.token_embedding.weight"][ids]
    e = rms_norm(embedded, weights["predictor.embedding_norm.weight"], eps)
    # `hidden` is already the target's post-output_norm hidden (llama.cpp, vLLM
    # and SGLang); the head only renormalizes it with hnorm.
    h = rms_norm(hidden, weights["predictor.hidden_norm.weight"], eps)
    # eh_proj reads the embedding first and the previous hidden second.
    hidden = np.concatenate([e, h], axis=-1) @ weights["predictor.fusion.weight"].T
    for layer, kind in enumerate(contract.mtp_mixer_types):
        prefix = f"predictor.block.{layer}."
        normalized = rms_norm(hidden, weights[prefix + "attention.norm.weight"], eps)
        if kind == "full_attention":
            branch = attention_reference(normalized, weights, prefix + "attention.", cache, offset)
        elif kind == "moe":
            branch = moe_reference(normalized, weights, prefix + "mlp.", contract)
        else:
            branch = dense_mlp_reference(normalized, weights, prefix + "mlp.")
        hidden = hidden + branch
    finalized = rms_norm(hidden, weights["predictor.output_norm.weight"], eps)
    return finalized, finalized @ weights["model.output.weight"].T, hidden


def predictor_cache():
    return [np.zeros((1, KV_HEADS, 32, HEAD_DIM), np.float16),
            np.zeros((1, KV_HEADS, 32, HEAD_DIM), np.float16)]


def mtp_reference(config, weights, previous, ids):
    """NumPy reference for one batched MTP head pass starting at position 0."""

    finalized, logits, _ = mtp_head(config, weights, previous, ids, predictor_cache(), 0)
    return finalized, logits


def mtp_chain_reference(config, weights, hidden, id_rows, *, chain_finalized=True):
    """NumPy reference for consecutive single-row draft steps.

    By default each step's final-norm output becomes the next step's hidden
    state, exactly like the runtime's depth chain. ``chain_finalized=False``
    chains the pre-final-norm hidden instead (the DeepSeek-V3 convention), which
    the negative test uses to prove step two reads the finalized value.
    """

    cache = predictor_cache()
    results = []
    for offset, ids in enumerate(id_rows):
        finalized, logits, pre_norm = mtp_head(config, weights, hidden, ids, cache, offset)
        results.append((finalized, logits))
        hidden = finalized if chain_finalized else pre_norm
    return results


def write_nemotron_fixture(path, config, weights, extra_components=()):
    from mfq.formats.assets import MODEL_CONFIG_ASSET, MODEL_GRAPH_ASSET
    from mfq.formats.header import FileHeader
    from mfq.formats.io import save

    components = [dict(kind="text", tensor_root="model", implementation="nemotron_h",
                       policy="decoder"), *extra_components]
    roots = ["model", *(component["tensor_root"] for component in extra_components)]
    topology = dict(text_layers=len(config["llm_config"]["layers_block_type"]))
    capabilities = ["text"]
    for component in extra_components:
        if component["kind"] == "predictor":
            # One predictor stage; the runtime applies every declared sub-block.
            topology["predictor_layers"] = len(
                config["llm_config"].get("mtp_layers_block_type") or ("attention",))
            capabilities.append("speculative_prediction")
        else:
            topology[component["kind"] + "_layers"] = 1
    graph = dict(
        schema_version=1,
        architecture="nemotron_h_omni",
        canonical_naming=dict(namespace="mfq.tensor", version=1, component_roots=roots),
        topology=topology,
        graph=dict(kind="causal_lm", backbone="nemotron_h"),
        components=components,
        capabilities=capabilities,
    )
    save(path, FileHeader(version=2, model_arch="nemotron_h_omni"),
         {**weights, MODEL_CONFIG_ASSET: json.dumps(config).encode(),
          MODEL_GRAPH_ASSET: json.dumps(graph).encode()})


def diagnostics_binary():
    value = os.environ.get("MFQ_NEMOTRON_H_NATIVE_TEST")
    if not value:
        pytest.skip("MFQ_NEMOTRON_H_NATIVE_TEST required")
    path = Path(value)
    if path.name.startswith("mfq-diagnostics"):
        return path
    sibling = path.with_name("mfq-diagnostics" + path.suffix)
    return sibling if sibling.is_file() else path.parent.parent / sibling.name


def run_fixture(path):
    return subprocess.run(
        [str(diagnostics_binary()), "--model", str(path), "--ctx-size", "32",
         "--check-nemotron-h"],
        text=True, capture_output=True, timeout=180)


def parse(process):
    raw = json.loads(next(line.removeprefix("nemotron_h_check ")
                          for line in process.stdout.splitlines()
                          if line.startswith("nemotron_h_check ")))
    assert raw.pop("architecture") == "nemotron_h"
    return {key: np.asarray(value["data"], np.float32).reshape(value["shape"])
            for key, value in raw.items()}


def run_mtp_fixture(path):
    return subprocess.run(
        [str(diagnostics_binary()), "--model", str(path), "--ctx-size", "64",
         "--check-nemotron-h-mtp"],
        text=True, capture_output=True, timeout=300)


def parse_mtp(process):
    raw = json.loads(next(line.removeprefix("nemotron_h_mtp_check ")
                          for line in process.stdout.splitlines()
                          if line.startswith("nemotron_h_mtp_check ")))
    assert raw.pop("architecture") == "nemotron_h"
    return raw


def mtp_fixture(tmp_path, mixers=LAYER_MIXERS):
    config, weights = model_fixture(mixers, mtp=True)
    path = tmp_path / "nemotron-mtp.mfq"
    write_nemotron_fixture(path, config, weights, extra_components=(MTP_COMPONENT,))
    return config, weights, path


@pytest.mark.parametrize("mixers", [
    LAYER_MIXERS,
    ("mamba", "mamba"),
    ("attention", "moe"),
    ("moe", "mlp", "attention"),
])
def test_native_complete_graph(tmp_path, mixers):
    config, weights = model_fixture(mixers)
    expected_hidden, expected_logits = model_reference(config, weights)
    path = tmp_path / "nemotron-graph.mfq"
    write_nemotron_fixture(path, config, weights)
    process = run_fixture(path)
    assert process.returncode == 0, process.stdout + process.stderr
    actual = parse(process)
    np.testing.assert_allclose(actual["full"][0], expected_logits[0], atol=2e-3, rtol=2e-3)
    np.testing.assert_allclose(actual["full_hidden"][0], expected_hidden[0],
                               atol=2e-3, rtol=2e-3)
    # b. prefill followed by single-token decode matches the prefill rows.
    np.testing.assert_allclose(actual["decode"][0], actual["full"][0][5:], atol=2e-3,
                               rtol=2e-3)
    # c. every chunk split of the same prompt matches one prefill.
    np.testing.assert_allclose(actual["chunked"][0], actual["full"][0], atol=2e-3, rtol=2e-3)
    # d. snapshot/restore and speculative rollback reproduce a fresh computation.
    np.testing.assert_allclose(actual["restored"][0], actual["continued"][0], atol=2e-3,
                               rtol=2e-3)
    for action in ("committed", "rejected"):
        np.testing.assert_allclose(actual[action][0], actual[action + "_reference"][0],
                                   atol=2e-3, rtol=2e-3)
    # e. batch>1 keeps every row equal to the single-sequence result.
    for row in range(actual["batch"].shape[0]):
        np.testing.assert_allclose(actual["batch"][row], actual["full"][0], atol=2e-3,
                                   rtol=2e-3)
    np.testing.assert_array_equal(actual["reset"][0], actual["full"][0])


@pytest.mark.parametrize("error", ["groups", "state", "activation", "latent",
                                   "layers", "topk", "conv"])
def test_native_rejects_inconsistent_config(tmp_path, error):
    config, weights = model_fixture(("mamba", "moe"))
    llm = config["llm_config"]
    if error == "groups":
        llm["n_group"] = 2
    elif error == "state":
        llm["ssm_state_size"] = 8
    elif error == "activation":
        llm["mlp_hidden_act"] = "silu"
    elif error == "latent":
        llm.pop("moe_latent_size")
    elif error == "layers":
        llm["num_hidden_layers"] = 5
    elif error == "topk":
        llm["num_experts_per_tok"] = 4
    elif error == "conv":
        llm["conv_kernel"] = 0
    path = tmp_path / "invalid.mfq"
    write_nemotron_fixture(path, config, weights)
    process = run_fixture(path)
    assert process.returncode != 0
    assert "nemotron_h_check " not in process.stdout


def test_native_rejects_declared_vision_component(tmp_path):
    config, weights = model_fixture(("mamba", "moe", "attention"))
    # The component inventory only needs one tensor under its root; the
    # adapter must still refuse to load a graph it cannot execute.
    weights["vision.patch_embedding.weight"] = np.ones((4, 4), np.float32)
    path = tmp_path / "components.mfq"
    write_nemotron_fixture(
        path, config, weights,
        extra_components=(dict(kind="vision", tensor_root="vision",
                               implementation="radio", policy="optional"),))
    process = run_fixture(path)
    assert process.returncode != 0
    assert "not implemented" in process.stderr
    assert "nemotron_h_check " not in process.stdout


def test_native_mtp_head_matches_reference(tmp_path):
    config, weights, path = mtp_fixture(tmp_path)
    process = run_mtp_fixture(path)
    assert process.returncode == 0, process.stdout + process.stderr
    actual = parse_mtp(process)
    previous = np.asarray(actual["previous"]["data"], np.float32).reshape(
        actual["previous"]["shape"])
    ids = np.asarray(actual["ids"]["data"], np.int64).reshape(actual["ids"]["shape"])
    expected_hidden, expected_logits = mtp_reference(config, weights, previous, ids)
    np.testing.assert_allclose(actual["head"]["data"], expected_hidden.reshape(-1),
                               atol=2e-3, rtol=2e-3)
    np.testing.assert_allclose(actual["logits"]["data"], expected_logits.reshape(-1),
                               atol=2e-3, rtol=2e-3)
    # A rejected draft row is fully discarded: the replay before the rollback
    # boundary and the continuation after it agree with one uninterrupted pass.
    np.testing.assert_allclose(actual["replayed"]["data"], actual["rollback"]["data"],
                               atol=2e-3, rtol=2e-3)
    # An accepted draft is committed as history: the committed rows match the
    # same rows of one uninterrupted predictor pass.
    np.testing.assert_allclose(actual["committed"]["data"],
                               actual["head"]["data"][4 * HIDDEN:7 * HIDDEN],
                               atol=2e-3, rtol=2e-3)


def test_native_mtp_target_hidden_is_post_output_norm(tmp_path):
    """The head consumes output_norm(h), not the raw backbone hidden."""

    config, weights, path = mtp_fixture(tmp_path)
    process = run_mtp_fixture(path)
    assert process.returncode == 0, process.stdout + process.stderr
    actual = parse_mtp(process)
    target_hidden = np.asarray(actual["target_hidden"]["data"], np.float32).reshape(
        actual["target_hidden"]["shape"])
    target_normalized = np.asarray(actual["target_normalized"]["data"], np.float32).reshape(
        actual["target_normalized"]["shape"])
    # a. the hidden captured for the predictor is the lm_head input itself.
    np.testing.assert_allclose(target_hidden, target_normalized, atol=0.0, rtol=0.0)
    ids = np.asarray(actual["ids"]["data"], np.int64).reshape(actual["ids"]["shape"])
    expected_hidden, _ = mtp_reference(config, weights, target_hidden[:, :6], ids[:, 1:])
    np.testing.assert_allclose(actual["head_from_target"]["data"],
                               expected_hidden.reshape(-1), atol=2e-3, rtol=2e-3)
    # b. negative: feeding the pre-output_norm hidden (output_norm's own input,
    # i.e. post_norm / weight up to the scale RMSNorm discards) must not match.
    pre_norm_equivalent = target_hidden[:, :6] / weights["model.output_norm.weight"]
    wrong_hidden, _ = mtp_reference(config, weights, pre_norm_equivalent, ids[:, 1:])
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(actual["head_from_target"]["data"],
                                   wrong_hidden.reshape(-1), atol=2e-3, rtol=2e-3)


def test_native_mtp_depth_chain_matches_reference(tmp_path):
    """Step 2 consumes step 1's own final-norm output as its hidden state."""

    config, weights, path = mtp_fixture(tmp_path)
    process = run_mtp_fixture(path)
    assert process.returncode == 0, process.stdout + process.stderr
    actual = parse_mtp(process)
    previous = np.asarray(actual["previous"]["data"], np.float32).reshape(
        actual["previous"]["shape"])
    ids = np.asarray(actual["ids"]["data"], np.int64).reshape(actual["ids"]["shape"])
    first, second = mtp_chain_reference(config, weights, previous[:, :1],
                                        [ids[:, :1], ids[:, 1:2]])
    np.testing.assert_allclose(actual["chain_first"]["data"], first[0].reshape(-1),
                               atol=2e-3, rtol=2e-3)
    np.testing.assert_allclose(actual["chain_second"]["data"], second[0].reshape(-1),
                               atol=2e-3, rtol=2e-3)
    # Negative: chaining the *pre*-final-norm hidden would change step 2.
    wrong = mtp_chain_reference(config, weights, previous[:, :1],
                                [ids[:, :1], ids[:, 1:2]], chain_finalized=False)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(actual["chain_second"]["data"],
                                   wrong[1][0].reshape(-1), atol=2e-3, rtol=2e-3)


def test_native_mtp_generation_matches_plain_decode(tmp_path):
    _, _, path = mtp_fixture(tmp_path)
    process = run_mtp_fixture(path)
    assert process.returncode == 0, process.stdout + process.stderr
    actual = parse_mtp(process)
    # b: the primary correctness guarantee - speculation never changes the
    # tokens greedy decoding emits.
    assert actual["greedy_mtp"] == actual["greedy_plain"]
    assert actual["cycles"] > 0
    # d: weights are loaded but the request switch disables speculation.
    assert actual["greedy_disabled"] == actual["greedy_plain"]


def test_native_text_only_load_ignores_present_mtp_weights(tmp_path):
    _, _, path = mtp_fixture(tmp_path)
    # The MTP component is declared and its weights exist; a text-only load
    # must degrade to the backbone instead of failing.
    process = run_fixture(path)
    assert process.returncode == 0, process.stdout + process.stderr
    assert "nemotron_h_check " in process.stdout


def test_native_rejects_declared_mtp_without_weights(tmp_path):
    config, weights = model_fixture(("mamba", "moe", "attention"))
    path = tmp_path / "missing-mtp.mfq"
    write_nemotron_fixture(path, config, weights, extra_components=(MTP_COMPONENT,))
    process = run_mtp_fixture(path)
    assert process.returncode != 0
    # The container validator and the component loader both name the missing
    # predictor instead of silently serving a text-only model.
    assert "declares predictor" in process.stderr and "missing" in process.stderr
    assert "nemotron_h_mtp_check " not in process.stdout


def test_native_rejects_incomplete_mtp_head(tmp_path):
    config, weights = model_fixture(("mamba", "moe"), mtp=True)
    # Only the fusion tensor survives; the norms and both sub-blocks are gone.
    weights = {name: value for name, value in weights.items()
               if not name.startswith("predictor.") or name == "predictor.fusion.weight"}
    path = tmp_path / "incomplete-mtp.mfq"
    write_nemotron_fixture(path, config, weights, extra_components=(MTP_COMPONENT,))
    process = run_mtp_fixture(path)
    assert process.returncode != 0
    assert "predictor" in process.stderr
    assert "nemotron_h_mtp_check " not in process.stdout
