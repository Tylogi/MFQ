# NINTv2 PLE row decoding on native Metal

Native Qwen3.8-Flash-Next PLE lookup accepts either the original E4M3 FP8
shards or packed NINT shards. NINTv2 uses the existing canonical wire format:
each embedding row has its own primary-code width `q` and subgroup-metadata
width `k`. This does not introduce a new dtype or change the quantizer's
allocation policy. Conversion remains opt-in through `--quantize-ple`.

Wire parsing and row gathering now live in the backend-neutral `NintRows`
layer; the Metal wrappers retain borrowed mmap access and the same kernel.
See [CUDA row decoding](nintv2-ple-cuda.md) for the CUDA range-backed path
and NVIDIA validation commands.

The architecture-neutral `MlxMappedNintRows` reader borrows the mmap and builds
compact cohort-rank checkpoints every 256 rows. It reads selectors at load
time but does not copy, unpack, or upload the full q/k payload. A lookup copies
only selected rows' packed q values, subgroup scales/minima, and FP16 neuron
anchors into a temporary batch. Rows from multiple shards, with different q/k
profiles and group sizes, can share one Metal dispatch.

`mfq_cpp_nint_mapped_row_decode` reads q/k from each row descriptor and computes
`neuron_scale * subgroup_scale * q_value - neuron_min * subgroup_min`.
The result is F16 by default; the shared operator also permits F32. Tail
columns, non-byte-aligned cohort streams and repeated IDs retain request order.
The caller owns the mmap for the lifetime of its borrowed row reader.

PLE shards must have identical logical shapes and must all be FP8 or all be
NINT. FP8 still applies its original shared scale. NINT stores that scale in
its quantized anchors and does not require the removed FP8 scale sidecar.
Hash multipliers, offsets and vocabulary sizes stay exact integer metadata;
key/value projections and recurrent convolution behavior are unchanged.

Validation includes actual Metal mixed-q/k and uniform packed-NINT decoding,
an independent scalar oracle, the existing resident NINT decoder, cross-shard
and tail cases, invalid inputs, and a `[2500012,160]` virtual table with
inaccessible unrequested payload pages. This establishes row-decoding
correctness and demand access, not full-model quantization quality or an
end-to-end speedup. The default PLE remains FP8: random-SSD latency and model
quality must be measured before changing that policy.

The native model-graph test writes matched small FP8/NINTv2 artifacts through
the Python codec and runs the real C++ Flash-Next graph. Prefill, incremental
decode, EOS boundaries and two-batch execution must produce identical logits;
mixing FP8 and NINT shards is rejected. Build `mfq-metal-nint-rows-test`, then
run `pytest tests/test_native_nint_ple.py` to exercise this integration gate.
