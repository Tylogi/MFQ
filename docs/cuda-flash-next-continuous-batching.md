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
