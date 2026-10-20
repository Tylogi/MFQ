# Cached projection GPU dispatch: design and cost, 2026-10-15

The implementation below is being evaluated on top of the measured 67.4138 tok/s API version. Reuse its saved reference; do not repeat old benchmark configurations.

## Why this is the next structural candidate

R17 profile: MoE 7.4653 ms/token, including resident arithmetic 2.5747, exposed GU wait/snapshot 2.1903, transfer GU .7433, transfer Down/reduction .7144, and later Down wait/scatter .4867. These intervals are sequential on the main GPU stream but the host and copy-engine intervals overlap them. Warm true expert payload is 33.7715 MB/token; the prior measured PCIe ceiling ~25 GB/s gives ~1.351 ms of bulk data time, not a whole-model latency prediction. Remaining wait cannot all be attributed to cache policy or data bytes.

Warm transfer classification is 2.486 experts/layer even after raw small metadata and weight fields hit GPU loan storage. Classifying cached complete experts as resident could move their arithmetic ahead of the CPU/DMA wait. The current `primary_residency_kind` reads only primary GPU maps. Merely changing CPU dispatch kinds would not work.

Both actual comparison runtimes use PCIe fraction 1; Strata is not hiding a CPU expert share in this comparison. Its natural-text API uses 32K context and shorter cold generation; MFQ uses 64K and 937 ordinary output tokens. Retain those limits when reporting comparisons.

## Feasible representation without changing weight layout

Existing math reads `cohort_by_expert` then a `MfePackedProjection`, whose `expert_local=nullptr` represents local expert zero. Append one compressed expert descriptor per expert/projection to the existing projection table. For 48 layers, 512 experts and 184-byte descriptors this costs about 13.55 MB per captured M shape. No expanded weights or large field-major cache arena is required.

The existing standalone `mfe_ffn_prepare.cu` can select, per routed expert, either the original primary descriptor or the cached single-expert descriptor in a mutable GPU cohort map. Arithmetic then uses the ordinary GPU descriptor path; primary hits need no extra fallback descriptor traversal. Keep immutable primary maps separately for selection. Do not modify `mfe_ffn_device.cuh` simply to recompile all formats (~26 minutes).

Per-shape activation pointers (fields 7/8), shared GU row offsets, NINT bounds and NVQ raw sizes must come from that runtime's validated templates. A cached pointer copied from a different M cell is insufficient.

## Publication and lifetime rules that must be implemented first

1. A new cache entry cannot become GPU-visible before its descriptor and weight copies complete. Current route IDs are published early, so a CPU valid bit set during allocation could race GPU prepare. Publish eligibility at the following route boundary, after the old consumer and descriptor snapshot finish.
2. Snapshot eligibility once in prepare, then write only routed GPU cohort indices. Arithmetic must not repeatedly read mapped validity bits. Duplicate routes use atomic or unique-owner updates.
3. Protect every current routed cache entry before any miss allocation. An eviction invalidates mapped eligibility before reusing its bytes; later-layer prepare must observe invalidations. A GPU mirror updated only next token would be stale.
4. Cached descriptor updates may be folded into the existing mapped-copy/snapshot packet, targeting all already-created M cells. No side-stream D2D backfill and no new mid-graph allocation.
5. Retirement after promotion and KV growth must invalidate eligibility before the next prepare, with reader fences preserved. Newly created M cells must initialize descriptors from existing cache entries before graph capture.
6. GPU prepare choice and CPU dispatch must agree. Complete valid cache hits become GPU resident; partial projections retain transfer dispatch. Cancel, CPU-quota, legacy prepare and explicit opt-out paths preserve current behavior.

Required evidence: quantify resident-versus-cached-transfer scheduling cost with production math and equal compressed bytes, then exact replay tests for repeated/mixed routes, cross-layer eviction, KV growth, GU-combined storage, M1/2/4/6 and cancellation. Only then run the new enabled API combination and compare against existing records.

## Implemented candidate

`MFQ_MOE_KV_GPU_RESIDENT` is enabled for eligible GPU-only native-DMA phased windows, alongside the three previously measured cache defaults. Explicit 0 disables the new classifier. Each M runtime appends geometry-only compressed-expert descriptors and retains an immutable primary cohort map. The existing standalone prepare kernel snapshots eligible field pointers into these GPU templates and updates only routed cohort indices. Math reads ordinary GPU descriptors; no expanded weight layout, side-stream D2D backfill, or extra graph node is introduced.

The shared mapped expert table holds three sets of eight raw field pointers plus one validity mask. New allocations enter a pending-publication list. The following reader boundary publishes them; eviction, retirement and KV growth invalidate them before reuse. Complete cached resident experts are pinned before any transfer misses can allocate. Partial cached experts retain the measured transfer path. Duplicate routes have one metadata writer. Per-M activation pointers and codebooks remain in the validated GPU templates. Combined Up field offsets are normalized in the publication observer.

The scalar version's isolated real-weight graph is 114.384 us/layer for ten fully cached experts, versus the saved 118.299 us cached-transfer probe and 107.168 us ordinary-resident probe. These are ready-flag scheduling probes without real host handshakes, not end-to-end speed. The old initial 989/567 us raw probe is anomalous and is not used as a cost estimate. A later vector-pointer timing run overlapped model unload and is excluded; do not rerun the unchanged configuration to manufacture a comparison. The vector implementation still has exact correctness coverage.

## Correctness and notification fix

Normal 168, serial two-layer 84, and adaptive native-exchange 28 exact fixture comparisons passed for the initial scalar implementation. Serial replay includes cache pressure and KV growth. After aligned 16-byte field loads, the serial fixture passed again. A separate real S2-L check covers 72 layer outputs: six layers, M1/2/6, shared/no-shared, fully cached and mixed primary/cache paths. All are bit-exact. These checks do not yet cover a real combined-GU cache fixture; the publication offsets are implemented but that configuration remains to be exercised.

Observer tests verify deferred publication, initial subscription to existing complete entries, eviction invalidation, one-time retirement invalidation, delayed revival, and KV growth. The existing field/geometry checks still pass.

Serial replay exposed a separate WDDM notification case: GU has DMA plus a driver-written ready flag, while Down has no DMA. The old finalizer only flushed the weight stream for pending Down DMA or host event polls; driver GU notifications set neither condition. Event polling completed, while driver notification stalled. The finalizer now flushes the submitted weight stream whenever called (fully empty phases already bypass it). The original driver-notification fixture then passed. No switch to host polling is required.

Original data are in `workspace/local-server/kv-gpu-resident-*-check-20261015.log`, `mfe-gpu-resident-s2l-check-20261015.log`, `mfe-gpu-resident-scalar-profile-20261015.log` and `cache-gpu-resident-api-build-20261015.json`. No real API speed is claimed before the new enabled measurement finishes.

## API evidence and continued implementation

The first enabled API combination is 59.5779 cold / 67.0848 warm tok/s, versus the saved 61.3559 / 67.4138 reference. Ordinary requests, outputs and token usage are identical. All 49 expert layers print an actual cached-resident dispatch. Natural MTP is 64.2973 with 766 tokens and 13/17 accepted; its trajectory differs from the 67.41 reference's MTP case, so it is not an isolated MTP comparison. The new feature remains implemented and enabled while its overhead is improved.

A per-expert generation plus copied-projection mask now reuses GPU snapshots until publication or invalidation changes the generation. A primary-to-cached projection change also refreshes fields not yet copied within that generation. Actual addressed-copy checks capture a graph, change a cached field to a new allocation, zero the old compressed payload, and replay: twelve real S2-L layer outputs (six layers, fully cached/mixed) remain exact. Normal 168 and serial 84 fixture comparisons pass. No unchanged performance baseline is rerun.

This implementation's API is 60.9961 cold / 66.1499 warm, same ordinary outputs and usage, with 49 actual cached-resident layers. Natural MTP is 64.0465, 766 tokens and 13/17 accepted. The warm ordinary result still does not beat the reference. Files: `speed-kv-gpu-generation-api-20261015.json`, `cache-gpu-generation-api-comparison-20261015.json`, `cache-gpu-generation-api-dispatch-20261015.json`.

The generation probe with the full service resident showed >1 ms in the final reduction, and slow whole graphs. After unloading that service, a separate explicitly checked empty-service configuration gives 107.734 / 109.371 us/layer for fully cached/mixed paths, with normal phase costs. This is evidence of interference with the loaded service, not proof of its exact WDDM cause. Neither the resident-service anomaly nor the earlier unload-overlapped timing is used to claim kernel speed. Independent performance probes must account for whole-device memory pressure; per-process CUDA free bytes are insufficient here.

The next change restores route ID publication at the beginning of prepare, allowing CPU planning to overlap the GPU cache lookup. A separate release flag in the same mapped allocation acknowledges completion of the GPU field snapshot. CPU fetch waits for that flag before any allocation, retirement or publication can mutate the table. CPU plan and GPU classifier read the same stable table, and cached readers remain protected before any miss allocation. No CUDA event or graph node is added. Serial 84 and adaptive native-exchange 28 exact fixture comparisons pass. A new full-model profile is running with this enabled implementation, using the saved R17 stage/traffic records as its reference.

## Early-route profile and later enabled API results

That profile completed as `speed-kv-gpu-early-route-profile-20261015`. It predates classifier reuse, the cached GU packet, partial cached GU, reader-fence reuse and shared scalar-gate fusion. Against the saved R17 profile, MoE is 7.46526→7.17566 ms/token; GPU GU wait 2.19033→1.7054, transfer GU 0.74330→0.4586, while prepare/gate is 0.67909→0.9376 and resident compute 2.57471→2.9352. Cached arithmetic moves into the resident interval. Net MoE change is about -0.290 ms, not the sum of just the improved phases. Profile throughput is 53.1550 cold / 60.2512 warm, and must not be substituted for the unprofiled API speed below.

Warm PCIe payload is 33.7808 MB/token, complete cached resident experts 95.7073/token, transfer-classified experts 0.491631/layer, host fetch 2.80204 ms/token. Saved Strata warm transfer classification is about 0.44/layer. Those traffic counts do not support blaming the remaining entire speed gap on an absence of cache hits. Stage boundaries and quantization formats differ.

Reusing the prepare classifier and cached GU packet gives enabled API 61.3437 cold / 67.4426 warm. The following partial-GU and reader-fence implementation gives 61.5309 / 67.5882. Ordinary requests, text and usage match the saved 67.4138 reference, and 49 layers report actual complete cached-resident dispatch. These small differences are not established stable gains. Natural-text MTP is respectively 64.3458 and 64.3824; acceptance/output trajectories differ from the 67.4138 record, so no isolated MTP attribution is made. All earlier changes remain enabled.

Partial GU preserves ready GU projection pointers when Down misses, protects those keys before miss allocation, and uses the existing early-GU arithmetic. A serial ten-route fixture retires two Down sources between replays to force this path: 84 exact cases/cancellation checks pass, with nonzero partial-GU and reader-fence counters. Separate real-weight checks pass 72 partial-GU outputs and twelve changed-address captured-graph outputs. Reader-fence reuse is restricted to the GPU cached-snapshot path with more than two routes and no outstanding forecast prefetch keys.

## Shared scalar gate in existing input preparation

The new candidate fuses the shared NINT8 scalar gate with warp input preparation, preserving original group/chunk reduction order, Half rounding and sigmoid. M1–6, GS48, one output row and eligible cached runtimes only; `MFQ_SHARED_GATE_PREPARE=0` disables it. It does not change the original compressed weights. Per-group terms and wrapping completion counters replace the separate scalar-gate graph node.

Real S2-L, six layers, M1/2/6, fully cached and mixed cases pass 144 changed-input captured-graph outputs plus the initial cases. The same check under Compute Sanitizer finishes with its PASS marker and `ERROR SUMMARY: 0 errors`; log `mfe-prepare-gate-s2l-check-20261015-memcheck.log`. The production normal pipeline fixture also passes after integration.

A new schedule comparison includes gate recomputation on every call; older cached probes computed that gate only once and are not used as its baseline. For shared M1 and 96 FFNs/graph, separate gate uses 1152 nodes and 109.9520 μs/layer; fused preparation uses 1056 nodes and 106.6880 μs/layer. This is a 3.264 μs local saving and one fewer kernel per eligible layer, not an API speed claim. `mfe-prepare-gate-s2l-new-schedule-bench-20261015.log` contains the data. The freshly linked service is loading for its first enabled API measurement; the measured 67.59 binary remains saved in `workspace/local-server/kv-cache-67-59-checkpoint`.

The first enabled API measurement has now completed: 61.8669744 cold / 67.9583043 warm tok/s, with ordinary requests, full text and usage identical to the saved reference. This is +0.55% warm over the previous 67.5882 result, still too small to call a stable whole-model gain from this single measurement. Natural MTP is 64.6851, 863 outputs and 9/17 draft tokens accepted; its differing trajectory prevents isolated attribution. Actual fused-gate messages and 49 cached-resident layers are saved in `cache-prepare-gate-api-dispatch-20261015.json`; source/executable hashes are in `cache-prepare-gate-api-build-20261015.json`. The service is idle with zero failed requests. The measured executable and build evidence are saved in `workspace/local-server/kv-cache-67-96-checkpoint`.

The warp state-scale LUT reuse prototype is retained separately in the bench directory. Its 432 exact real-weight cases pass; candidate-only timings do not exceed the already saved matched-schedule controls beyond their drift. No unchanged timing was rerun and it is not presented as a production improvement.

## Published-route ordering replaces a redundant stream wait

Historical activation review found no explicit `MFQ_MOE_WINDOW_ROUTE_FENCE` in the old route-fence profile, and no actual skipped-event counter. The old binary's default is not directly available, so it is an unresolved activation question rather than proof the candidate definitely ran. Its old negative conclusion is withdrawn. See `historical-activation-audit-20261015.md`.

The current cached-resident window now uses its already observed graph route publication and snapshot acknowledgement to establish prior-reader completion before fetching. Only that eligible path defaults to skipping the redundant `cudaStreamWaitEvent(weight_stream, compute_done)`; explicit 0 restores it, and standalone/legacy defaults remain. A first-actual-skip log is emitted per Cell, after checking that the wait would otherwise have been submitted. Empty packets returning before this branch are not counted as skipped waits.

Normal 168, serial ten-route partial-GU/cache-pressure 84, and explicit-off 168 exact fixture cases pass. The enabled fixtures assert that an event was actually skipped; the off fixture asserts no skip log. The first enabled API gives cold 62.0548213 / warm 68.5291940, versus saved 61.8669744 / 67.9583043 from gate fusion. Ordinary requests, output text and usage match; +0.30%/+0.84% from one measurement is not claimed stable. No old baseline or unchanged configuration was rerun. Actual dispatch includes 271 per-M Cells with a skipped event, 292 fused-gate Cells, and 49 cached-resident pipeline reports; these are first-use coverage counts, not per-token traffic counts.

Natural MTP is 65.05597 with 766 outputs and 13/17 accepted. It differs from the immediately prior MTP trajectory, so no isolated event-fence attribution is made. Saved cached-GU/reader output matches exactly at 64.38242, but two intervening changes separate those versions. The running service is idle with zero failed requests, and the measured executable is in `workspace/local-server/kv-cache-68-53-checkpoint`. Files use `speed-kv-route-fence-api-20261015` and `cache-route-fence-api-*` prefixes.

## Combined profile of the measured 68.53 build

`speed-kv-route-fence-combined-profile-20261015` profiles that exact executable and production source hashes. All six cache/gate/fence flags are explicitly enabled, with first-actual-dispatch assertions. The native profile configuration is compared to the already saved early-route profile; the only environment differences are explicit current optimization flags and the dispatch-evidence requirement. No old profile or unprofiled API configuration was rerun. Ordinary outputs and token usage match the old profile.

Profile throughput is 53.8168 cold / 61.4444 warm, with timestamp instrumentation. It does not replace the 62.0548 / 68.5292 API measurements. The existing stage parser reports 1192 valid profile replays; user decode traffic counters cover 255 cold and 936 warm windows. The same aggregation method is retained for the historical comparison.

| Stage | Earlier early-route ms/token | Current combined | Existing Strata weighted reference |
| --- | ---: | ---: | ---: |
| PLE + attention GR | 1.55206 | 1.50001 | 1.51236 |
| GDN | 2.99012 | 2.98986 | 2.63707 |
| FFN GR + router | 1.62166 | 1.60573 | 1.97942 |
| MoE | 7.17566 | 6.98778 | 5.98280 |
| QSA | 1.69985 | 1.69364 | 1.86236 |
| Head + final GR | 0.76758 | 0.76202 | 0.52236 |
| Other | 0.19772 | 0.19721 | 0.04000 |

Prepare/shared gate is 0.93755→0.64195 ms, resident arithmetic 2.93522→2.94816, GU readiness 1.70540→1.78782, transfer GU 0.45862→0.45971, Down readiness 0.47893→0.47523 and Down/reduction 0.58343→0.59423. Net MoE reduction is 0.18789 ms, rather than the isolated 0.29560 ms preparation saving. Earlier preparation exposes some later wait.

Warm PCIe payload is 33.76820 MB/token, cached-resident experts 95.74359/token and transfer-classified experts 0.4921875/layer, almost unchanged from the earlier profile. Host plan/fetch/join are 0.37009 / 2.77965 / 0.21960 ms/token. Cache-hit behavior alone does not explain the remaining speed gap.

As a coarse comparison only, MFQ resident arithmetic plus preparation is about 3.5901 ms, while Strata shared/quant plus resident arithmetic is about 2.6035. MFQ GU/Down waits plus CPU wait total about 2.3069, versus Strata waitA/waitB/waitCPU about 2.6840. These different fusion boundaries do not establish a one-kernel cause, but they favor investigating arithmetic/dispatch before another generic attempt to remove PCIe waiting. Weight formats and the references' cold/warm proportions still differ.

Machine-readable stage/traffic reports: `kv-route-fence-combined-{stages,traffic}-20261015.json`. Profile source identity: `cache-route-fence-combined-profile-build-20261015.json`. Production remains the measured 68.53 implementation.
