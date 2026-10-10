# Studio resource tiers

The overview displays an OpenAI SDK base URL ending in `/v1`. This is a display
and clipboard URL; Studio's management API base URL is unchanged.

`GET /api/v1/runtime/instances` includes an optional `memory` breakdown per
instance. Studio aggregates ready/busy instances, not just the selected model:

| Tier | Measurement |
| --- | --- |
| Resident experts and dense weights | `resident_weight_bytes`, model weight allocations |
| Resident KV and prefix cache | `kv_bytes`, live context state plus resident prefix bytes; context groups and resident prefix blocks |
| SSD-streamed experts | `ssd_expert_bytes`, backing record payloads when streaming is enabled |
| SSD-streamed PLE tables | `ssd_ple_bytes`, backing row-table record payloads when streaming is enabled |

SSD payloads are not added to resident memory. Cumulative bytes read, allocator
reuse caches and process RSS are not substituted for these measurements.
Missing telemetry is `null`, not zero. Busy workers can retain the last reported
breakdown when their nonblocking telemetry lock is unavailable.

The native Metal worker aggregates counters registered by cache and storage
components, independently of model names or backbone-specific runner methods.
MFE format/geometry probes do not register SSD backing. A record enters that
count only when a layer switches to streaming or actually pages its experts;
only those active records are counted after a partial weight offload. Their
cached RAM pages remain in dynamic weight residency. SSD backing denotes file
payloads, not an additional RAM allocation, and cache hits do not duplicate it.
New backbones reusing these components inherit telemetry; a new storage/cache
component registers its own resource lifetime once. Refresh does not evaluate
devices, copy arrays or read files. Static weight residency is sampled
after load staging is released, before generation; dynamic MFE weight residency
is added separately. Preallocated expert arenas belong to weight residency.
Live cache bytes include attention K/V, index caches, GDN recurrent state and
PLE convolution state, including predictor caches. They describe retained cache
storage capacity, not just visible token rows. Temporary speculative work is
not included. Disk-only prefix blocks/contexts are excluded from resident counts.
Workers without these counters explicitly show unavailable breakdowns. Partial
aggregates retain known values as lower bounds and identify missing models.

Each model gets one color shared by all four tiers. Registration follows load
order: existing accent blue, then red, green, yellow and additional colors.
Unloading a model does not recolor the remaining models during the view's
lifetime. The legend is present even for a single model.

The resident-weight track uses the detected usable inference memory limit
(`runtime_memory_effective_budget_bytes`, falling back to the configured runtime
budget on older servers). Its right-hand label displays numeric used / limit
values. The KV/prefix track uses that same limit minus the resident weights of
all ready/busy instances, clamped to zero. Resident segments represent utilization
of their respective limits, not shares normalized to a full bar. On overcommit,
the bar is full and retains all model colors; the numeric limit is not increased.
SSD segment widths remain each model's share of that tier's reported bytes.
Missing measurements are excluded and marked as unavailable. An unknown memory
limit, or unknown weight residency when calculating KV capacity, is displayed as
`--` and does not produce a misleading full resident bar.

Loading bars appear in the overview, loaded-model list and local checkpoints.
Native workers report unique tensor records whose sources have been prepared
(read, mapped or parsed for streaming), followed by runtime finalization. These
are preparation steps, not elapsed-time estimates or physical SSD bytes read.
Aliases share one step and metadata assets do not count. Unsupported workers
show indeterminate progress; only a ready runtime completes the bar.
