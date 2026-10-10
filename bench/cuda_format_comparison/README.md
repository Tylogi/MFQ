# CUDA format benchmarks

The native CUDA test build includes benchmarks for one expert projection,
ten routed experts, the fused MFE FFN, gated residual projections, shared
gate/up projections, and BF16 matrix-vector multiplication. These tools use
released MFQ weights supplied by the caller; weights are not distributed.

`cases.example.tsv` lists representative S2-L and V4-XS formats and shapes.
Copy it to a local manifest and set its model column to the first MFQ shard
of each checkpoint. Pass that manifest explicitly to the format runners.

```sh
cmake --build build/cuda --target mfq-cuda-format-bench mfq-cuda-format-moe-bench mfq-cuda-real-mfe-bench
build/cuda/cuda/mfq-cuda-format-bench cases.tsv results
```

For an independent llama.cpp comparison, configure this directory with
`-DLLAMA_SOURCE=<llama.cpp-checkout>`. The comparison executable quantizes
the same decoded MFQ weights to the selected GGML format. This compares
execution cost and does not establish equal model quality.

GPU event timings exclude model loading, reference quantization, graph
construction, and CPU dispatch. The runners rotate independent weight
addresses larger than L2 and report seven warmed timing samples. Effective
bandwidth counts the selected packed GPU fields and excludes cached
codebooks; it is not a hardware DRAM counter.

The runtime retains compact NVQ payloads by default. The enabled paths
include whole-group reads, decoder reuse, mixed-format prefill cohorts,
NINT affine reduction, packed GR projections, GDN prefill pipelining, and
QSA fusion. Expanded E8/D4 execution layouts remain optional and disabled.

Synthetic regression targets cover original compressed upload, byte signs,
format bounds, routed NINT/NVQ1, projection composition, and graph reuse.
The full FFN pipeline tests additionally cover memory limits, registered
RAM, cache exchange, prefill/decode transitions, and repeated ownership.

`generate_mfe_gu_register_probe.py` generates an optional local occupancy
probe snapshot. Generated device snapshots and raw results are not part of
the source distribution.
