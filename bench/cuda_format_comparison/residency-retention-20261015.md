# CUDA expert residency: longer routing memory

At the service's four-token adaptation interval, a new expert used once every
12 tokens has a maximum decayed score of `1 / (1 - 0.7^3) = 1.522`, below the
existing admission threshold of 2. Keeping 90% of heat per round lets recurring
experts accumulate evidence across longer reuse gaps. This is a property of the
ranking rule, not proof that this threshold explains the entire Strata gap.

Change the CUDA default retention from 0.7 to 0.9 and log the actual retention,
interval and projection mode when the residency manager starts. At interval 4,
the half-life changes from 7.77 to 26.32 tokens. The environment override,
admission/gain thresholds, projection exchanges, prefill heat seeding, weight
layouts, and existing copy/decoder paths remain unchanged. This change does not
implement dual-window scores, byte-aware admission, or format-pool resizing.

## Controlled Python measurement

RTX 3090 Ti, S2-L, 1,386 prompt tokens, 1,024 output tokens; temperature 0.7,
top_p 0.8, top_k 20, seed 1234; FP16 KV, 64K context limit, SSD KV disabled,
RAM preallocated, prefix reuse and MTP disabled. Timing excludes everything
before the first output token. Both 0.9 outputs exactly match the saved 0.7
output. Reuse the existing 0.7 measurement instead of rerunning it.

| Metric | 0.7 first request | 0.9 first request | 0.9 second request |
| --- | ---: | ---: | ---: |
| Decode tokens/s | 44.8431 | 47.3198 | 49.4630 |
| Demand expert PCIe payload MB/decode token | 191.505 | 164.436 | 146.825 |
| CPU window fetch ms/decode token | 10.652 | 9.462 | 8.756 |
| Residency join ms/decode token | 1.254 | 1.092 | 0.779 |
| Committed projection exchanges | 35,416 | 28,808 | 21,613 |

The matched first request improves throughput by 5.52% and reduces demand
expert payload by 14.13%. There is no second-request 0.7 result for this exact
configuration. CPU work, DMA and GPU execution overlap; do not add the phase
times. Demand payload excludes cache-maintenance exchanges.

## Service checks and limits

The rebuilt service reports `retention=0.9 interval=4 projection_heat=1` with
all eight previously required decoder/transfer dispatches observed. Zero
request failures; FP16 KV and the 64K context limit remain active.

| Workload | Current tokens/s | Saved reference tokens/s |
| --- | ---: | ---: |
| Natural text, complete-sequence warm cache, 937 outputs | 69.0549 | 70.0393 |
| Python MTP, 1,536 outputs | 54.6045 | 54.9236 |

Natural-text outputs match the reference exactly. Its result is 1.41% below
the saved reference; no natural-text speedup is claimed. The MTP result accepts
782 of 788 drafts (99.24%); its prior cache history differs from the reference,
so no MTP improvement is claimed. An earlier natural-text request had only a
256-token warmup and is not the complete-sequence warm-cache comparison.

The default direct-exchange regression passed 154 scenarios plus 12 repeated
scenarios, covering exact RAM/GPU weight bytes, partial backup/refill failures,
map/host/publication rollback, reader exclusion, delayed publication and
cancellation. The broader warm-residency fixture was also attempted with the
full service environment and did not pass its packed-buffer/workspace counter
assertions. Those unsuccessful checks are not reported as passes; no fixture
or production exchange logic change is included to bypass them.

Source/runtime contracts and automatic CUDA residency tests: 72 passed locally.

The initial benchmark runner incorrectly required one nonempty text event per
Chinese output token. That request completed normally, but the runner stopped
afterward. Its native-log result was recovered separately and was not rerun.
Subsequent service checks retain complete metrics without that assumption.

Runtime SHA256:
`90ef375600967868deb2cdebcd5a313d4fff0ec735134276c5a764a94c38417a`.
Machine-readable measurements are in `residency-retention-results-20261015.json`.
Local raw artifacts are under `workspace/local-server/python-retention09-20261015`,
`retention09-service-20261015`, and `retention09-full-sequence-warm-20261015`.
