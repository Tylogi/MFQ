# CUDA GDN chunk prefill

The shared GDN operator in `mfq/kernels/cuda/gated_delta_net.cu` dispatches
scalar log gates to `gdn_prefill_chunk.cu`. Model adapters use the existing
operator entry points. CUDA SM80 or newer uses warp-level FP16
`mma.sync.m16n8k16` with FP32 accumulation. Operands use independent power-of-two
scales before entering Tensor Cores; products restore the combined exponent.
Token gate exponentials are computed once in shared memory and reused across
channels.
Persistent state, log gates, beta and outputs remain FP32. Q/K/V may be FP32,
FP16 or BF16. The recurrent state is retained in FP32 registers between chunks;
its scratch representation does not impose the FP16 exponent range.

For one 64-token chunk, let `G_i` be the local inclusive sum of log gates and
`S` the state before the chunk. With rows indexed by tokens, define:

- `L_ij = beta_i exp(G_i-G_j) dot(k_i,k_j)` for `j<i`, and zero otherwise.
- `M = (I+L)^-1`, `U = M (beta V)`, and `W = M (beta exp(G) K)`.
- `Delta = U-W S`.
- `O_i = [exp(G_i) q_i S + sum(j<=i) exp(G_i-G_j) dot(q_i,k_j) Delta_j] / sqrt(D)`.
- `S_next = exp(G_last) S + sum(i) exp(G_last-G_i) k_i outer Delta_i`.

Chunk-local gate scans, KKT, triangular solves and WY preparation run across
all chunks in parallel. A second kernel advances state across chunks and
computes outputs with matrix products. The triangular solve has dependencies
within each local row; recurrent state is updated once per chunk, rather than
once per token. Gate differences are formed locally without inverse decay
factors or a sequence-wide cumulative sum.

Widths 32, 64 and 128, grouped and tiled key-head mappings, ordinary and
transposed state storage, optional initial state and in-place state updates
share this implementation. State CTAs own disjoint value columns. The largest
preparation CTA requests 56,640 bytes of dynamic shared memory; state
propagation requests 79,744 bytes at width 128. Dispatch checks the device's
actual opt-in shared-memory limit. Workspace is capped at 256 chunks (16,384 tokens) per request:
a scaled FP16 `[B,Hv,chunks,64,D]` W buffer, an FP32 U buffer of the same
shape, a scaled FP16 `[B,Hv,chunks,64,64]` attention buffer, and local gate
prefixes with the two packed-matrix exponents. These are the same FP16 operands
that enter MMA; packing them once avoids repeated casts and global reads.
Longer prompts reuse these buffers in consecutive windows, carrying the full
FP32 state between windows. No per-token recurrent states or quadratic
full-sequence matrices are stored.

The window uses at most a quarter of free VRAM after allocating the required
output and state. If even one chunk cannot fit, uniform and variable-length
calls use the recurrent operator without modifying input state first.
`MFQ_GDN_PREFILL_CHUNK_WINDOW` can lower the chunk count; zero disables the
transient workspace and selects this fallback.

`gdn_prefill_varlen_cuda` accepts padded `[B,H,T,D]` requests and host lengths
in `[0,T]`. Requests have independent states; output padding is zero and a
zero-length request preserves its initial state. This CUDA-specific helper
requires SM80 or newer. Metal and HIP continue using their existing recurrent
operators; serving interfaces and model behavior are unchanged.

`MFQ_GDN_PREFILL_CHUNK=0` selects the previous recurrent implementation for
uniform requests. `=1` forces eligible chunk calls above eight tokens.
Automatic dispatch starts at 4,096 tokens per operator call. Runtime prefill
slicing is selected separately; a smaller slice retains recurrence even when
the complete prompt is longer. Smaller projected prefills retain
the existing FP32 recurrence: subsequent normalization and FP16 projection can
amplify small dot-product rounding differences. The forced benchmark still
measures the chunk path below that threshold. Single-token decode and
verification windows of up to eight tokens keep the existing short kernels.
An explicit `MFQ_GDN_PREFILL_COLUMNS` retains its previous dispatch unless the
chunk option is explicitly enabled. Per-key KDA gates continue using the existing recurrent kernels.

The native `mfq-window-attention-test` exposes `--gdn-prefill-chunk-check`,
`--gdn-prefill-chunk-varlen-check`, `--gdn-prefill-chunk-graph-check`,
`--gdn-prefill-chunk-bench` and `--gdn-prefill-chunk-shapes-bench`. Correctness
uses the existing output/state tolerance (`atol=2e-4`, `rtol=5e-3`). The benchmark
alternates baseline and chunk trials, omits warmup and reports
median CUDA-event latency, including operator workspace allocation. Run GPU
commands under the shared GPU lock.

`bench/gdn_prefill_chunk.py --binary <mfq-diagnostics> --gpu-lock <lock>` generates
a temporary single-layer model with hidden size 128, GDN heads 16/48, width
128, PLE and two routed experts. It alternates the two GDN paths at all four
prompt lengths, discards two warmup pairs and reports six paired wall-time
samples by default. The candidate is explicitly forced below the automatic
threshold too. Generated model weights are removed after the run.
