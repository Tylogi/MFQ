# CUDA continuous batching benchmark

Build the CUDA runtime as usual, then run the separate `mfq-bench` executable:

```sh
mfq-bench continuous-batching --model /path/to/model.mfq \
  --ctx-size 16384 --prefill-chunk-size 2048 \
  --prefill-tokens 8192 --baseline-tokens 8 --gen 16 --reps 5
```

`--config /path/to/config.json` is available for models requiring an external
configuration. The other values shown are defaults. The workload requires a
dense Qwen3.5 model supported by the CUDA continuous batcher and a GPU.

The tool runs two requests: A decodes while B enters in chunks. It checks the
outputs against serial greedy generation and reports A's baseline and contended
inter-token latencies, B's time to first token, throughput, and scheduler
metrics. The first run warms the scheduler; `--reps` controls measured runs.
