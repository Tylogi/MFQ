# CUDA KV memory policies

The CUDA Flash-Next/QSA worker exposes `kv_memory` in model load requests and
per-model context settings. It is independent of `kv_quantization`; FP16 remains
the default. Metal keeps its existing dynamic allocation and storage controls.

```json
{"kv_memory": {"placement": "auto", "allocation": "preallocated"}}
```

| Placement | Raw K/V storage |
| --- | --- |
| `vram` | Preallocated GPU storage; reject a context that cannot fit. |
| `ram` | Preallocated mapped host memory, accessible to CUDA kernels. |
| `auto` (default) | Budgeted VRAM hot KV and preallocated RAM cold KV; reject insufficient combined capacity. SSD KV is disabled. |
| `auto_ssd` | Budgeted VRAM hot KV, RAM cold KV, then SSD spill. RAM may be preallocated or allocated as KV grows; SSD storage is created on demand. |

The service's automatic VRAM budget includes the entire target indexer cache
plus a smoothly interpolated fraction of raw KV: 100% at zero context, 60% at
128K, 30% at 256K and 10% at 1M. An explicit `gpu_budget_bytes` must cover the
indexer cache. Unused KV capacity can hold extra experts, which are evicted as
KV grows. VRAM and RAM tiers exchange cold rows without retaining a full
duplicate in RAM. `ram_preallocated` defaults to true; disabling it requires
`auto_ssd`. The service checks disk capacity before admitting SSD-backed KV.

Studio's model loading policy includes a collapsible Context management
component. An empty default context means 32768 tokens, capped by the model's
supported limit. Automatic settings disable KV quantization and SSD KV and
enable preallocation. Manual settings distinguish sparse-attention RAM/SSD
controls from other models' VRAM preallocation. Defaults are saved per server
in the current browser and resolved against model/backend capabilities when
loading. Explicit API KV settings take precedence over these defaults.

Indexer rows, pooled index keys, recurrent state and compute scratch remain on
the GPU. Capacity admission accounts for dense weights, the minimum expert
working set and compute reserve. Expert residency is reduced by the concrete KV
reservation so prefill transfer buffers retain headroom. Windows additionally
retains shared-memory budget for mapped plans and pinned transfer buffers.

`allocation: "dynamic"` is experimental. The virtual-memory helper is adapted
from local Strata under its MIT license. It reserves stable addresses for the
configured context, initially maps up to 16K tokens of raw KV, and grows in 8K
steps rounded to the driver's allocation granularity. Existing KV is not copied
when pages are added. Indexer storage is currently preallocated.

GPU kernels resolve RAM cache misses while decode/MTP graphs retain stable
addresses. Selected rows remain resident across attention calls; queries larger
than the hot window can read cold RAM rows without an unbounded wait. SSD-backed
attention uses stable GPU staging and performs required host I/O outside graph
replay. Indexer and recurrent state remain on the GPU. Prefix restore and
rewritten speculative positions invalidate stale resident rows. KV precision
is unchanged (FP16 unless explicitly configured).

Native CLI callers that do not supply the service's tier-budget settings retain
the legacy 64K threshold and 32K hot-window policy, configured through
`MFQ_KV_AUTO_STREAM_MIN_CONTEXT` and `MFQ_KV_RESIDENT_CELLS`. Unified-memory
devices do not use that legacy discrete-memory policy.

Long-prompt prefill also budgets sequence-sized residual and FFN injection arrays
before choosing a compute tile. When borrowing expert storage, the two transfer
ring slots use separate allocations so their combined size need not fit one
contiguous region. If the second slot cannot fit, prefill uses one slot.

Dynamic allocation currently requires GPU-resident raw KV (`vram` or `auto`).
Growth headroom is retained in the expert budget; direct physical-page exchange
with expert arenas is not implemented. Additional concurrent sessions are
admitted against available GPU memory; explicit RAM placement uses mapped RAM
for each session. Unsupported model backends do not advertise these controls.

The default CUDA allocation policy is preallocation. Capacity failures without
SSD use `kv_memory_insufficient`; the Studio message offers enabling SSD KV or
reducing context length. Settings are persisted per model after a successful
reload, and a failed reload restores the previous settings.

Local Strata reference `fb58e0dbc8399662c0e47c76578c6e878b14f6cf` has a
4096-token native CLI default. The README's recommended interactive setup instead
selects context from the model and memory budget (typically 128K for Flash-Next
on a 24 GiB GPU). From 64K, when RAM permits, that setup keeps raw KV in RAM
with `--kv-resident 32768`; it normally uses INT8 KV above 8K. MFQ adopts the
long-context placement without changing the user's context or KV precision.
Strata's expert-cache budget accounts for model, KV and workspace costs.

Native regression coverage includes four placements, RAM/SSD spill ordering,
generation parity, reset, prefix restore, MTP, continuous batching and stable
addresses/data across virtual-memory growth.

The Flash-Next service retains compatible CUDA decode graphs across request
reset and prefix restoration when KV addresses are stable. GDN/PLE state is
cleared or restored in place, while QSA pooled keys are refreshed after prefill
and restoration. Expert-budget changes and switching to independent batch
execution invalidate graphs before their transfer buffers are released.

Loading prepares ordinary single-token graphs, target verification widths 1–4
and predictor widths 1–4. Short adaptive catch-up uses the same four-row windows.
The largest predictor
is prepared first so smaller shapes share its transfer buffer. Diverse synthetic
token IDs and two additional replay sweeps exercise the captured widths during
loading. All synthetic model and predictor state is cleared afterwards. This work adds
to model loading rather than interrupting the first generated response.
Long-prompt prefill bounds its compute tile by remaining VRAM while these graphs
are retained; `runtime_prefill_chunk_budget` reports the requested and selected
tile sizes. The logical prompt chunk and configured context capacity do not
change. New, unprepared graph geometries can still incur a first-use capture.
MTP continuations of up to 128 new tokens reuse the existing target windows;
short predictor priming uses windows of up to four tokens. Predictor graph
execution retires prefill expert images before replay. First replay costs after
each request's prefill are excluded from adaptive draft-depth timing.

CUDA Flash-Next also runs up to 64 synthetic output tokens through the complete
MTP generation loop after graph capture, using temperature 0.7, top-p 0.8 and
top-k 20. This exercises stochastic sampling, draft-history
replay and commit/rollback paths before accepting requests. It clears synthetic
KV, generation metrics and adaptive-depth policy afterwards. The extra loading
work is bounded by the model context; `MFQ_QWEN4_WARMUP_MTP=0` disables it.

Whole-layer prefill asynchronously records each target layer's last 32 routed
tokens into a reusable GPU buffer. One combined D2H copy retrieves the samples
at the decode boundary, avoiding tiny per-layer transfers that compete with
prefill expert streaming. Before long MTP predictor priming,
after the target workspace loan has returned, the residency manager prepares
experts using this real prompt tail. Earlier prompt chunks replace their samples
rather than accumulating an entire long prompt's counts. The tail has a four-token
half-life (about six token-equivalents), so a long prompt does not dominate
subsequent decode heat. Usual transactional publication,
rollback and memory limits still apply. This moves an initial exchange into the
reported prefill time. `MFQ_MOE_PREFILL_RESIDENCY=0` disables the observation, and
`runtime_moe_prefill_residency` reports its exchanged projections and elapsed time.

Adaptive MTP calibration revisits speculative widths after the ordinary-decode
baseline. CUDA Flash-Next starts with one draft, three baseline cycles and a
second shallow replay before trying wider verification, including re-entry after
ordinary decoding. Other backends retain
their existing initial order. A fresh CUDA policy repeats the width sweep once
more after initial residency adaptation; inherited continuation policies keep
the shorter calibration. Several cold transfer/cache cycles can survive graph warmup; a single
slow first sample must not suppress a width until the multi-second exploration
interval. Calibration uses the fastest valid sample for each width, followed by
the existing steady-state estimator. Maintenance cycles and widths shortened by
the output/context limit do not consume calibration or probe samples. Acceptance
measurements still include those cycles, and consistently costly speculation
can still select ordinary decoding. After the post-baseline shallow replay,
CUDA defers wider warmup entries whose estimated throughput cannot beat the
baseline, or whose cold projected cost exceeds the startup latency budget.
Deferred widths remain eligible for subsequent exploration.
CUDA parks speculation after six valid consecutive losing cycles following
calibration, avoiding another costly narrow probe in a short response. Other
backends retain the sixteen-cycle exit threshold. Maintenance remains excluded.
Three final scalar samples refresh the baseline after residency adaptation, so
width selection does not compare warm speculation with a cold ordinary step.

A fresh CUDA policy initially caps verification at one draft token. During the
first 48 emitted tokens, widening requires the observed cycle time, scaled by
the next verification width, to fit a 60 ms budget. This is a startup estimate,
not a hard inter-token latency guarantee: actual expert routes and transfers
can change between windows. Inherited continuation policies keep throughput
selection without this cold-start cap. Maintenance samples cannot authorize
widening, and the bounded startup period prevents permanent under-calibration.

During ordinary-decode calibration, predictor history is replayed every four
tokens. Parked speculation uses sixteen-row batches, split into four-row tiles
to amortize expert-exchange boundaries without accumulating a long replay at
request completion. These captured graphs omit final mixing, the
vocabulary projection and token selection. These history-only widths 1–4 are
also captured at load. Only the final tile of a history merge computes the
draft output head. Re-entry combines at most 128 pending history
rows with the current row, avoiding a separate one-row predictor launch while
preserving token positions and committed KV history.
`MFQ_MTP_HISTORY_EARLY_FLUSH=0` restores the previous 512-row flush threshold.

CUDA verification graphs preserve GDN recurrent/convolution state and PLE
convolution/ngram context at every speculative prefix. Partial rejection selects
the accepted checkpoint and truncates QSA KV. Accepted draft tokens do not run
through the target model a second time. Eager execution keeps its existing
rollback/replay behavior. Additional recurrent checkpoints are included in the
runtime's cache memory accounting; packed expert weights retain their layout.

For conversational measurements, `bench/cuda_mtp_conversation_sampling.py`
uses open-ended Chinese questions and stochastic target/draft sampling
(temperature 0.7, top-p 0.8, top-k 20 by default). It refuses temperature zero,
saves the generated text, per-delta timestamps, acceptance/depth/parking metrics,
and the sampling parameters. Load the model afresh before measuring a cold first
turn. `--rounds 2` includes the continuation in the same conversation.
The default background is synthetic repeated English text of roughly 8K tokens;
`--background` selects a document instead. Counting and near-100% acceptance
measurements do not represent this conversational workload.

CUDA Flash-Next completes its one-token target bootstrap before publishing the
initial two-token batch. It preserves the prompt-only prefix snapshot before
bootstrap and previews the first token's EOS/stop behavior without changing the
real output or sampling state. Penalties and constraints use prospective copies.
This moves necessary startup work into actual TTFT and avoids a cold bootstrap
stall immediately after the first text; it does not remove that compute cost.
`MFQ_MTP_EARLY_BOOTSTRAP=0` restores separate first-token publication. Other
backends retain their publication order.

Fresh single-device, layer-major Flash-Next prefill can borrow expert-cache tails
while retaining decode graphs. This follows local Strata's `plan_lend` / `lend`
design: count compute workspace and the layer transfer ring together, then
temporarily stream the experts whose GPU storage is borrowed. MFQ preserves the
original allocations and slot books. It saves one compressed RAM copy of the
borrowed tails and builds temporary cold expert maps; two reusable host staging
buffers upload their weights with each layer. RAM prefix storage yields capacity
before these backups are allocated.

Temporary CUDA allocations reuse free slices of the borrowed regions on the
prefill compute stream. Releasing a tensor returns its slice for later phases,
so sequential scratch does not accumulate. Persistent state and graph allocations
bypass the loan. On completion, the final residual is detached, transfer readers
are synchronized, all borrowed tensors are released, and original weights are
restored at the same addresses. Cancellation and exceptions use the same cleanup.
Decode rejects an outstanding loan if cleanup cannot complete.

`runtime_prefill_loan` reports borrowing and return. `MFQ_PREFILL_BORROW=0`
disables this path; `force` is a diagnostic override used by small native tests.
The current compute estimate remains conservative rather than Strata's exact
`Prefill::bytes_needed` formula. Free memory, transfer growth, fragmented expert
fields and the minimum resident expert set can still require a smaller compute
tile. Cached long continuations and other model/prefill paths retain their
existing allocation policy. Ordinary expert-cache resizing remains a separate
operation that invalidates graphs and the contiguous layer-prefill layout.

MTP teacher-forced prompt priming also borrows workspace when its predictor
expert image and compute tiles exceed available memory. A single predictor layer
uses one transfer slot; multiple layers may overlap two slots.
Gate/Up and Down transfer images are allocated separately during a single-layer
loan so they can use separate expert regions rather than requiring one oversized
contiguous allocation. Persistent
predictor positions bypass the loan. A short priming tail returns the loan before
using decode graphs, and cancellation releases it with the speculative session.
Borrowing and restoration time is included in prefill timing.

Speculative generation publishes each committed token batch before computing
the next draft chain. Consumer/network delay is excluded from the next MTP cost
sample. Cancellation between batches updates committed predictor history without
running another speculative chain. The CUDA worker reports early cycles above
55 ms as `runtime_mtp_slow_cycle`, separating draft, verification and commit time
and reporting target expert transfer bytes. `MFQ_TRACE_MOE_SERVE` additionally
reports decode-window reset, graph launch, service and final wait times.

Expert residency exchanges retain their full adaptation rate. Transactions may
mix incoming weights already held in the GPU transfer cache with uncached RAM
weights: only uncached incoming fields need to be preserved before victims are
copied directly into their final RAM locations. When the existing GPU transfer
wire has sufficient payload space, it holds those incoming fields instead,
eliminating the CPU backup copy without allocating more VRAM. Only uncached
fields occupy this compact payload; cached fields leave no unused gaps. Its control header
is preserved; the expert fence retires previous consumers and the next decode
waits for the transaction before reusing it. Insufficient wire space uses pinned
RAM backups. Cached incoming fields refill
from GPU storage. Publication and rollback preserve both kinds of original
weights. GPU-wire exchanges pipeline H2D preservation and D2H victim backups
on separate streams in groups of up to 8 MiB. A group fence protects each RAM
source until its original weights have been preserved; cached incoming fields
can begin their victim backups immediately. Refill and publication wait for all
backups, and rollback drains both streams before restoring bytes.
Large victim fields retain direct D2H DMA. Fields up to 64 KiB inside the
registered RAM arena share one mapped-write kernel after the preservation
fences. This removes small per-field DMA submissions while large transfers
can still overlap the graph tail. Fields outside that arena or beyond descriptor
capacity keep their individual DMA path. `MFQ_MOE_RESIDENCY_SMALL_BACKUP_BYTES=0`
disables this batching; its default limit is 65536 bytes. The exchange-wait log
includes total and batched backup bytes/field counts.
`MFQ_MOE_RESIDENCY_DUPLEX=0` retains the serial transfer path. One reusable pinned
image also supplies expert-map uploads. One mapped-copy kernel publishes all
maps after the previous window retires, avoiding a separate DMA submission for
each small map. Both its descriptors and payload remain alive through completion
and rollback.

Prefix snapshot admission counts occupied KV rows and recurrent state, excluding
unused KV capacity and speculative graph checkpoints. An unsuccessful initial
MTP snapshot is retried at request completion, rather than after every token.
MTP emits a prompt-state boundary after predictor priming and before appending
the first generated token. Initial prefix snapshots run at this boundary, so
their latency is included in TTFT and excluded from the decode interval.
This corrects timing attribution; it does not by itself reduce inference time.
Parser-held trailing text is flushed before the final prefix snapshot. Cache
storage and cleanup still finish before the terminal event; they no longer hold
back the last visible characters of an otherwise completed response.

CUDA expert transfer arenas pack per-projection fields while retaining stable
graph addresses. Preload copies pinned host weights directly into final expert
slots and releases temporary staging after transfer completion. Large MTP
prompt histories can remain in RAM and upload bounded predictor tiles; the
predictor budgets its compute workspace before borrowing expert capacity.

Grid vision weights remain in RAM by default and move to VRAM for each encoder
stage. Its workspace estimate includes decoded projection weights and MLP
activations. Fused tanh GELU avoids retaining a chain of full-sized temporaries.
GLM-5.3-Flash uses the same staging policy for its RMSNorm/SwiGLU tower and
channel-first downsampler, with sequential text positions for image/video
prompts. `MFQ_CUDA_VISION_RESIDENT=1` retains the vision tower on the GPU.
Windows image/video requests support the authenticated binary tensor-file transport
used by the Python server, with file ownership, shape and token validation.
