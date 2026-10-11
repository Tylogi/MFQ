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
