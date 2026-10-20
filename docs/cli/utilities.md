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

```shell
uv run mfq wt2-reference --model model-hf --dataset wikitext-corpus \
  --output reference.bin
```

## `mfq voice-runtime-check`

Verify that the optional MiniCPM-o voice output dependencies import on this
host: ONNX Runtime, the tokenizer and audio stacks, and the CosyVoice modules.
The command prints the first missing dependency and exits nonzero on failure.

```shell
uv sync --extra minicpmo45
uv run mfq voice-runtime-check
```

`mfq inspect` is reserved for future use and is not implemented yet.
