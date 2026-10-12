# Flash-Next CUDA continuous batching

Enable an explicit request capacity with `mfq serve --max-requests-per-runtime N`
or native `mfq-runtime --continuous-batching N`. Flash-Next supports capacities
from 1 to 32 on one CUDA device. The managed server's default remains one request.

Each request owns its GDN recurrent state, PLE convolution and ngram history,
QSA caches, token counts and logical position. Pending prefills run in bounded
chunks. Eligible decode requests share GR, routing and FFN computation, including
the expert transfer and decode pass. Paused requests do not advance their state.
Cancellation releases only the cancelled request's state.
Eligible rows wait for their peers to schedule the next decode, so requests
with different prefill lengths form a batch instead of alternating single-row
decodes. A request paused by output backpressure does not hold up other rows.

The decode window captures CUDA Graphs for the active request group. A change in
membership or cache capacity rebuilds the graph. QSA allocations grow with the
actual context; selecting a large context limit does not allocate that entire
limit for every slot. Request-owned KV and recurrent state are included in the
runtime's KV memory metrics.

Before growing caches, the runtime checks actual available VRAM, including the
transient expansion workspace, and can evict prefix hot-cache entries. A prefill
capacity failure rejects that request. If the active decode group cannot grow,
the newest eligible request receives a context-capacity error and existing
requests retain their state. Unexpected CUDA failures still fail the backend.
This does not resize the resident expert cache or automatically enable KV offload.

Explicit continuous batching currently runs ordinary token decoding. Batched
requests do not use MTP or persistent prefix reuse. Explicit session IDs and
multimodal requests remain exclusive. The default single-request path retains
MTP and prefix caching. Configured QSA KV offload uses the eager batch path.

Run the state check with the native diagnostics executable:

```shell
mfq-diagnostics --model model.mfq --ctx-size 512 --check-continuous-batching
mfq-diagnostics --model model.mfq --ctx-size 512 --check-continuous-batching --continuous-batching-check-requests 32
```

The check compares outputs against serial generation, exercises paused rows,
decode cancellation, partial-prefill cancellation and subsequent request reuse.
The larger check holds all requests until their prefills finish, then decodes
them together across the 256-token cache-capacity boundary. Small native model
fixtures include QSA, GDN, PLE and routed experts. The shared execution test also
injects prefill and decode-capacity failures to verify surviving requests, and
checks staggered request scheduling. The stdio chat-client integration test
compares simultaneous generations with serial output and verifies that the
decoder actually processes multiple tokens per batch.

## Allocator and cache growth

CUDA Graph scratch uses a separate asynchronous device pool from resident
weights and reusable quantization workspaces. Request GDN state is allocated
on first use in a resident pool. Retiring a graph trims its unused pages;
pool telemetry includes every live and retired pool. Pool handles are reclaimed
when their remaining tensors are released.

QSA keeps one KV layout while a graph is active. It releases the previous
sequence layout after preparing graph KV, and reconstructs that layout only
when leaving the graph. Batches above 16 requests grow graph capacity in
128-token increments after their initial allocation; smaller batches retain
the previous doubling policy. The configured context limit remains a ceiling,
with no full-context allocation at admission.

Eligible GDN layers process independent requests in one decode core launch.
A small address table points to each request's existing convolution, recurrent
state and counters; recurrence stays in those allocations. One subsequent
kernel commits convolution rows after all readers finish. Membership changes
rebuild the table with the graph. No second full batch of recurrent state is
allocated. `MFQ_GDN_INDEPENDENT_BATCH=0` restores the per-request core calls.
This dispatch requires batched attention projections and supported fused GDN
geometry. Unsupported layers retain their per-request path.
Request decode counters are prepared after prefill, before graph recording,
and are included in the growth reservation. `MFQ_TRACE_GDN_BATCH=1` reports
state eligibility, projection dtypes and actual batch core selection.

Quota errors during prefill or graph capture reject the affected request
without poisoning the backend. Other CUDA failures retain their existing
fatal-error behavior. Native tests inject failures after transfer submission
and during graph recording, then check subsequent execution and state.

For 17–32 rows, large NINT vocabulary projections read packed weights through
the online Tensor Core path. Other projections keep their existing GEMM path.
`MFQ_NINT_COMPRESSED_LARGE_BATCH=0` disables this dispatch; `head` selects the
default, and `1` enables it for other eligible projections as an experiment.
Decoded expert weights are not kept as persistent expanded allocations.

## Local validation, 2026-10-12

S2-L on an RTX 3090 Ti, 4096 context limit, 2048-token prefill chunk,
32-request capacity, 10.66 GiB expert residency and 22.5 GiB process VRAM limit.
The 32-request case follows two 1024-token MTP requests in the same process,
generates 256 tokens per request, and crosses the 256-token KV boundary.

| Case | Completed tokens | Total throughput | Decode graph throughput |
| --- | ---: | ---: | ---: |
| 32 requests, 256 tokens each | 8192 | 87.55 tok/s | 100.01 tok/s |
| 32 distinct prompts, 24 tokens each | 768 | 43.06 tok/s | 96.24 tok/s |

Both cases complete without errors and match their serial text references.
Total throughput includes prefill and graph capture; graph throughput measures
CUDA window replay only. These are single-run observations, not an isolated
A/B speedup claim. The distinct-prompt case uses a separate serial reference
for each request. Additional validation exercises 16 staggered arrivals and
checks ordinary decoding and MTP before and after large batches.
After both 32-request cases, 1024-token ordinary decoding measured 50.30 tok/s
and MTP measured 74.05 tok/s from streamed output after the first token.
Both continuations match the serial text reference. Cold MTP is slower;
these measurements describe the warmed process and do not establish a speedup.

With indirect GDN enabled under the same configuration, the 32 × 256 case
completed all 8192 tokens at 89.20 tok/s total and 101.71 tok/s graph throughput.
The distinct-prompt case measured 98.83 tok/s graph throughput; 16 staggered
requests completed all 2048 tokens at 111.94 tok/s graph throughput. Ordinary
decoding and MTP afterwards measured 50.40 and 73.27 tok/s from streamed output.
All outputs match their serial references. Independent-run variations should
not be treated as isolated speedup measurements.

The S2-L graph shrinks from 13,438 to 10,054 nodes, including a reduction from
3,200 to 896 copy nodes. A same-process GDN primitive comparison at 32 rows
measured 442.03 µs for independent launches and 289.92 µs for indirect batching.
Output, convolution and recurrence are byte-exact. Tests exercise changing
inputs and state address order at 2, 4, 6, 8, 16 and 32 rows with 4, 2 and 1
column tiles. The model fixture also asserts that 32 rows enter the new core.

Short GDN verification sequences use two columns at length 2 and four at
lengths 3–8 for eligible D128 layers. `MFQ_GDN_SHORT_COLUMNS=0` restores the
previous short-sequence dispatch; an explicit `MFQ_GDN_PREFILL_COLUMNS` value
retains priority. Single-token decode and long prefill dispatch are unchanged.
Same-process primitive measurements reduced length-2 recurrence from 7.17 to
6.51 µs and length-6 recurrence from 14.68 to 13.15 µs. Validation covers 360
exact multi-column cases, changing graph inputs, recurrence state, prefill to
decode transitions and the full native window test.

With this dispatch, two warm 1024-token S2-L MTP requests measured 73.95 and
74.57 tok/s from streamed output; ordinary decode measured 50.19 tok/s.
A subsequent 16 × 128 request batch completed all 2048 tokens at 93.18 tok/s
total and 119.54 tok/s graph throughput. All outputs match the reference.
These observations do not establish an isolated whole-model speedup.

Compact E8 projections retain the immutable JSC state-to-bank map in four
previously unused descriptor bytes. The map is prepared on the CPU while
loading weights and follows each shared codebook into expert slices and
transfer descriptors. This avoids one dependent metadata lookup per group;
weights, codebook storage and descriptor size stay unchanged. Zero maps retain
the previous lookup path, including valid single-bank codebooks.
`MFQ_MFE_NVQ_BANK_MAP=0` selects the previous path when constructing FFN views.

Real S2-L FFN graph checks cover 252 changing-input cases at 1, 2, 3, 4, 5, 6
and 8 tokens, shared experts, repeated routes and invalid routes. Output and
active hidden values match byte for byte; the fixture verifies the actual
compact descriptors contain their codebook maps. Six native metadata fixtures
also verify every state and expert slice, and eight NVQ request-batching cases
pass. Same-process complete FFN timings show small gains or variations around
zero, usually below 1%; these are not a whole-model speedup claim.
The complete-model regression finishes 32 × 256 tokens without errors at
101.64 tok/s graph throughput. Subsequent ordinary decode measures 50.00
tok/s and warm MTP measures 75.32 tok/s from streamed output; the earlier two
warm MTP runs measure 73.65 and 73.94 tok/s. All continuations match the
reference, including the separate serial oracles for 32 distinct prompts.
