# Utility commands

The smaller `mfq` commands beyond [build](build.md), [serve](serve.md),
[quantize](quantize.md), [calibrate](calibrate.md), and
[solve-ew](../quantization/ew-joint-solver.md).

## `mfq optimize-layout`

Rewrite a packed MFQ model into native execution layouts. Use it when an
artifact was produced for a different execution path or an older packer.

| Option | Meaning | Default |
|---|---|---|
| `--input PATH` | Source MFQ model. | Required |
| `--output PATH` | Optimized MFQ model. | Required |
| `--overwrite` | Replace an existing output. | Off |

```shell
uv run mfq optimize-layout --input model.mfq --output model-native.mfq --overwrite
```

## `mfq wt2-reference`

Generate official WT2 reference logits from original HF weights. The output
feeds KL evaluation through `mfq-eval kl --kl-base`.

Supported architectures are Qwen3.5 (dense and MoE), Gemma4, and
GLM-5.3-Flash (`glm5_next`). GLM reference generation requires native CUDA.
It reads the original FP8 block-scaled and floating weights, runs the text
layers with the runtime's four-stream mHC state, and excludes vision and MTP.
Each window starts with fresh attention/recurrent state. Tokenization uses
the checkpoint's tokenizer and special-token settings without a chat template.
At context 512, logits at positions 256..510 predict targets 257..511.

| Option | Meaning | Default |
|---|---|---|
| `--model PATH` | Original HF checkpoint. | Required |
| `--dataset PATH` | WT2 evaluation corpus. | Required |
| `--output PATH` | Reference logits file. | Required |
| `--context-size N` | Evaluation context size. | `512` |
| `--chunks N` | Corpus chunks. | `8` |
| `--backend {auto,metal,cuda}` | Execution backend. | `auto` |
| `--device NAME` | Device; must match the selected backend. | `mps` or `cuda:0` |
| `--resident` | Keep the model resident instead of layerwise streaming. | Off |
| `--runtime PATH` | Native CUDA `mfq-runtime` executable. | Search `PATH` |

```shell
uv run mfq wt2-reference --model model-hf --dataset wikitext-corpus \
  --output reference.bin
```

CUDA generation hands one chunk of float32 logits to the compressed writer
at a time. It keeps the existing `_logit3_` format and exact float32 target
log probabilities; logits are not rounded to float16 for the handoff.
Layerwise mode keeps one text layer's weights on the GPU at a time and stores
all chunks' hidden states on disk, preserving their shape and runtime dtype.

For `N` chunks, context `C`, vocabulary `V`, hidden width `H`, residual streams
`M`, and `S = C - C // 2 - 1` scored positions, reserve:

- Reference: `16 + N * (12 + 4*C + 4*S + 2*S*(2*ceil(V/2) + 4))` bytes.
- Hidden scratch: `4*N*C*H*M` bytes (float32 storage).
- Logits handoff: `4*S*V` bytes, independent of `N`.
- Token input: `8 + 4*N*C` bytes, plus a small JSON manifest.

For GLM-5.3-Flash at context 512 (`V=154880`, `H=4096`, `M=4`), this is
78,993,920 bytes per reference chunk, 33,554,432 bytes per hidden-state chunk,
and 157,977,600 bytes for the reusable logits handoff. At 580 chunks, peak
disk use is about 65.44 GB (60.94 GiB), including the 45.82 GB reference.
Scratch is placed beside the output and removed on success; the Python
command also cleans it up on failure. Allow filesystem overhead and free
space beyond these payload sizes. See the [GLM evaluation recipe](../glm53f-evaluation.md).

## `mfq voice-runtime-check`

Verify that the optional MiniCPM-o voice output dependencies import on this
host: ONNX Runtime, the tokenizer and audio stacks, and the CosyVoice modules.
The command prints the first missing dependency and exits nonzero on failure.

```shell
uv sync --extra minicpmo45
uv run mfq voice-runtime-check
```

`mfq inspect` is reserved for future use and is not implemented yet.
