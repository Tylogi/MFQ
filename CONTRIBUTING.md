# MFQ development rules

The [runtime architecture](docs/architecture.md) describes the
architecture as it is implemented today, including its transitional limits. The
target refactor architecture is [refactor.md](docs/refactor.md), the
migration order and exit gates are [plan.md](docs/plan.md), and known defects
are tracked in [bugs.md](docs/bugs.md); non-refactor follow-ups remain in the
workspace [TODO](../docs/todo.md). Existing backend code is not a precedent for
duplicating shared behavior.

Rules below are split into what binds work **now** and what the **target**
requires. Where the two differ, follow the current rule while migrating a
consumer, and never add new code that the target forbids.

`Runtime = Transport + Scheduler + Engine`. Runtime applications compose the
three components. Transport converts wire messages to scheduler commands;
Scheduler owns admission, cancellation decisions and bounded request outboxes.
The shared Engine owns physical execution, models and caches. Execution follows
`schedule -> bounded step -> typed results -> publish events`.

## Rule zero: reusable code must not be architecture-bound

Anything that can be shared by more than one model architecture **must not**
be implemented in `models/<architecture>/` or hidden behind an
architecture-named entry point.

- Runtime placement and cost rules live in `engine/weights/`, task collection
  in `engine/executor/`, and logical prefix/QSA state in `engine/state/`.
  Host byte storage, disk reservations and encoded-row leases belong to
  `backends/cpu/storage/`; measured CPU work belongs to CPU execution.
  Only the transitional `mfq-core` target remains for K4. Schemas and row readers
  already belong to `formats/`, model input to `model_source/`, and vision
  geometry to `engine/model/common/vision/`. Do not add new code to `core/`.
- `mfq-model-graph` parses only persisted schema. `mfq-model-source` opens inputs,
  normalizes metadata and retains bounded readers; neither may depend on core,
  Engine or model execution. Keep alias maps in `model_source/compat/`; public
  source headers expose only layout values. Headers sit beside implementations.
  Normalize legacy dtype spellings in input parsing; operator dispatch receives
  canonical IDs. Tests and synthetic benchmarks normalize their fixture input.
- Packed format headers and frozen codebooks live in `csrc/formats/`
  (`mfq-formats`), without a model-source, Engine or backend dependency.
  Compiled range readers in `formats/quant/` use `mfq-quant-readers`; keep them
  independent of core and model loading, and keep kernels on the header target.
  CUDA native kernels link this header target directly; they must not link
  `mfq-core` or use its include directories.
- File ranges, batched reads and I/O statistics live in `csrc/components/io/`
  (`mfq-io`), with headers beside the implementation. It links only host
  thread primitives and must not depend on model loading, formats, Engine
  or device operations. Host memory copying belongs in `backends/cpu/ops/`.
- Logical scheduling, cancellation, deadlines and output backpressure belong
  in `csrc/scheduler/`; wire protocols belong in `csrc/transport/`.
- Cross-backend request execution, tokenization, sampling semantics, physical
  batch policy, session/cache lifecycle, metrics and MTP acceptance/commit rules
  belong in `csrc/engine/`.
- Engine request/media values, sampling parameters, grammar cursor and
  `StepSequence` live in `csrc/engine/include/`. Sampling has one definition in
  `sampling_params.h`; device sampling adapters include that narrow contract.
  Native allocators include `resource_error.h` directly without core.
- Shared model configurations and forward definitions live in
  `csrc/engine/model/<family>/`, with shared layer compositions in
  `csrc/engine/model/common/`. Their non-template implementations compile in
  `mfq-engine`; backend adapters and tests opt into the private model include
  path. Family equations, positions and topology are defined once. The target
  composes complete backend operators; [refactor.md](docs/refactor.md) defines
  that execution chain and
  [bugs.md](docs/bugs.md) B28 tracks the current binding chain
  (`CausalModelBase → CudaCausalOps → native model`) as debt, not as a rule to
  reproduce.
- Reusable mathematical operations and packed kernels belong in
  `csrc/backends/<backend>/ops/` and `csrc/backends/<backend>/kernels/`.
  The shared CUDA `.cu/.cuh` sources already live in
  `csrc/backends/cuda/kernels`; `mfq/kernels/cuda/` holds only the Python
  wrappers, `mfq_cuda.cpp` and the `_ext.py` manifest, which compiles a subset
  of those same native sources. There is no pending kernel move.
- CUDA `native/` owns tensors and contexts, `execution/` owns runtime context
  and expert task submission, `storage/` owns weight materialization
  and physical cache storage, and `engine/` is the sole composition root for
  family selection, runtime components, batching and CUDA Graphs.
  `models/<family>/` supplies native operator/state bindings, with reusable
  bindings in `models/common/`. CUDA model code must not include CUDA `engine/`,
  and `models/common/` must not include a concrete family. These directories
  are relative to `csrc/backends/cuda/`. In the target, backend storage is the
  physical implementation that a shared StateStore/WeightStore composes, and the
  backend `engine.cpp` only assembles; see
  [refactor.md](docs/refactor.md) and plan §1.4 R4.
- CLI parsing and application composition belong in backend `commands/` and
  `apps/`, compiled into the corresponding executable.

The second architecture that needs an existing behavior is a mandatory
extraction point: move the behavior to the appropriate shared layer before
adding the new adapter. Do not copy, rename, or lightly modify an existing
model loop. A model name in a reusable state machine, sampler, cache
lifecycle, metric type, precision dispatcher, multimodal pipeline, MTP loop,
or serving policy is an architectural defect. Model-family forward definitions
are model-owned; public request lifecycles are not.

An exception is allowed only when checkpoint semantics genuinely differ. The
implementation must document that invariant next to the code and include a
test that would fail if the special path were replaced by the shared one.
Performance preference, deadline pressure, and “only one model uses it today”
are not exceptions.

## Dependency and header boundaries

Transport depends on scheduler contracts. Scheduler depends on the shared
Engine interface. Shared Engine/model code must not include backend headers or
branch on backend identity. Backends provide native types and compile-time
operations; do not add a universal type-erased tensor/KV layer, callback Engine,
or per-backend generation loop. Backend operations must not publish request
events or receive token emitters, cancellation callbacks or transport objects.

CUDA exposes `include/cuda_engine.h` for Engine options and construction.
Other headers live beside their owning implementation or in `kernels/` for
kernel declarations. Internal consumers include those headers explicitly;
internal include paths stay PRIVATE in CMake. Do not add forwarding headers
for old paths or expose implementation directories through a PUBLIC include.

`mfq-cuda-runtime` and `mfq-paged-prefix-cache` build independently of transport.
`MFQ_BUILD_RUNTIME_COMMUNICATION=OFF` removes communication modules and the
serving executable; it does not disable backend, tokenizer, cache, diagnostics
or evaluation targets. Only serving applications link backend and transport.
`MFQ_BUILD_CUDA_RUNTIME` is the matching backend switch: only it enables the
CUDA language, finds the CUDA Toolkit, and adds the CUDA targets, tests and
benchmarks, so the common engine and its CPU tests configure, build and run
without invoking a CUDA compiler or linking a CUDA library.

Inside `backends/cuda`, a cross-directory include is written as
`"<dir>/<name>"` (`#include "ops/moe/moe.h"`, never `"moe.h"` or `"../ops/moe/moe.h"`);
the build no longer adds the layer directories to the include path, so the bare
form cannot compile. The directories form a DAG from lowest to highest:
`native`, `kernels`, `ops`, `storage`, `models`, `engine`. A file may include
its own directory or a lower one. `apps/`, `commands/`, `diagnostics/`, `eval/`
and `tests/` are top-level consumers. `tests/test_cuda_include_graph.py`
enforces the include form and the DAG, and `tests/test_layer_independence.py`
keeps the shared layers free of backend and model-family names.

A layer is grouped by concept, not by file count: when several files in one
directory share a name prefix, the prefix becomes a subdirectory
(`kernels/nvq/device.cuh`, not `kernels/nvq_device.cuh`).  The include rule is
unchanged, the path is still rooted at the layer directory.

`engine/<module>` (`model`, `requests`, `batch`, `state`, `weights`, `executor`, `text`,
`src`) is PRIVATE in `engine/CMakeLists.txt`, and the backends and tests add
those directories to their include path and include their headers bare. That is
the only way a backend drives the shared generation loop, so it is allowed for
`backends/` and `tests/`; the boundary check lists the two temporary old-model
geometry consumers explicitly. Other consumers are forbidden: `core/`, `transport/`,
`scheduler/`, `models/` and `components/` must not include an engine private
header. `engine/` never includes a backend header, and inside `engine/` the
request table, Planner, Store rules and text layer never reach the Executor
driver. A backend's `ops/`, `kernels/` or `execution/` code never includes the
shared Executor or Store headers; `storage/` may, because it is the physical
StateStore and WeightStore. The engine header graph is a DAG and no private
header is orphaned. `tests/test_engine_include_graph.py` checks all of this, and
carries one small fixture per rule so a rule that stopped firing fails the test
by itself.
`tests/test_engine_include_graph.py` enforces all four.

There is no `runtime/` layer and no `Runtime` type. Composition and lifetime
live in the application executable (`apps/`, `commands/`); a word that names no
responsibility is not a module.

## Canonical internal representation

Raw Hugging Face, GGUF, or historical MFQ tensor names may appear only at the
import/compatibility boundary. Conversion maps them once into canonical MFQ
names. Runtime operators consume canonical roles such as attention
projections, MLP gate/up/down, mHC, predictors, and multimodal components; they
must not branch on a source repository's spelling.

Compatibility readers must be isolated under an explicit `compat/` or
`legacy/` boundary. New writers emit only the canonical schema. Do not add a
silent alias merely to avoid fixing an importer.

## Backend parity

Backends must share the **public semantics**: request lifecycle, sampling
semantics, component names, canonical roles and the meaning of a model family's
equations. They must not share a fake `supported = true`.

Actual capability is per backend and per feature, and is stated item by item in
the [runtime support matrix](docs/runtime/support.md), not assumed equal across
backends. CUDA and Metal may use different kernels, physical layouts and float
results, and one backend may lack a mechanism the other has (for example
interleaved batching and CUDA Graphs today). Architecture support is registered
from components present in the model, not from scattered per-model booleans.
Optional Vision and MTP components default to enabled when both the graph and
weights provide them; missing weights or an explicit user override disables them.

## Change checklist

Before submitting a runtime or quantization change:

1. Identify the owner: core, file I/O, transport, scheduler, shared Engine/model,
   backend operation/storage, importer, or application.
2. Search all architectures and both backends for an equivalent implementation.
3. Extract reusable behavior before extending it; keep model adapters thin.
4. Test changed behavior and protect critical ownership rules under `tests/`.
   Source-boundary checks and compilation do not establish numerical accuracy
   or performance; use real model weights for those claims.
5. Per commit, run the focused checks for what you changed: the affected native
   test targets and the affected Python suites. Run the full native CTest suite
   and the broader relevant Python integration only at a stage exit or when the
   change touches shared runtime dependencies. The authoritative test contract -
   which entry points cover which behavior, what environment they need, and what
   counts as a pass - is
   [plan.md §4 全阶段测试与验收](docs/plan.md#4-全阶段测试与验收); build scope
   follows its §1.5 G4 table, so a tool cold build stays conditional. Build a
   target before running `ctest` against it, and treat a CTest skip
   (`SKIP_RETURN_CODE 77`) or a `model_branch=skipped` line as not verified. For
   build-boundary changes, also build and test with
   `MFQ_BUILD_RUNTIME_COMMUNICATION=OFF`, and with both `MFQ_BUILD_CUDA_RUNTIME`
   and `MFQ_BUILD_METAL_RUNTIME` off when the change touches the common engine.
6. Do not publish generated files, local paths, credentials, model weights, or
   disposable benchmark artifacts.

Code review must reject architecture-local duplication even when it works for
the first model. The maintenance cost is paid by every model added afterward.
