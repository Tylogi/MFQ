# MFQ development rules

The workspace [runtime architecture](../docs/architecture.md) defines ownership
for CUDA and shared runtime/model code. This guide applies those rules to the
repository. Pending migrations remain in the workspace [TODO](../docs/todo.md);
existing backend code is not a precedent for duplicating shared behavior.

`Runtime = Transport + Scheduler + Engine`. Runtime applications compose the
three components. Transport converts wire messages to scheduler commands;
Scheduler owns admission, cancellation decisions and bounded request outboxes.
The shared Engine owns physical execution, models and caches. Execution follows
`schedule -> bounded step -> typed results -> publish events`.

## Rule zero: reusable code must not be architecture-bound

Anything that can be shared by more than one model architecture **must not**
be implemented in `models/<architecture>/` or hidden behind an
architecture-named entry point.

- Backend-neutral value contracts, schemas, canonical names and model sources
  belong in `csrc/core/`.
- Logical scheduling, cancellation, deadlines and output backpressure belong
  in `csrc/scheduler/`; wire protocols belong in `csrc/transport/`.
- Cross-backend request execution, tokenization, sampling semantics, physical
  batch policy, session/cache lifecycle, metrics and MTP acceptance/commit rules
  belong in `csrc/engine/`.
- Shared model configurations and forward definitions belong in
  `csrc/models/<family>/`; shared layer compositions belong in
  `csrc/models/common/`. Family equations, positions and topology are defined
  once and composed with backend operations.
- Reusable mathematical operations and packed kernels belong in
  `csrc/backends/<backend>/ops/` and `csrc/backends/<backend>/kernels/`.
  Shared CUDA kernel implementations still reside in `mfq/kernels/cuda/`
  pending the source/package move recorded in TODO.
- CUDA `native/` owns tensors and contexts, `storage/` owns weight materialization
  and physical cache storage, and `engine/` binds shared execution to device
  state and CUDA Graphs. `models/<family>/` supplies native operator/state
  bindings, with reusable bindings in `models/common/`. These directories are
  relative to `csrc/backends/cuda/`.
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

CUDA exposes `include/mfq/cuda/engine.h` for Engine options and construction.
Other headers live beside their owning implementation or in `kernels/` for
kernel declarations. Internal consumers include those headers explicitly;
internal include paths stay PRIVATE in CMake. Do not add forwarding headers
for old paths or expose implementation directories through a PUBLIC include.

`mfq-cuda-runtime` and `mfq-paged-prefix-cache` build independently of transport.
`MFQ_BUILD_RUNTIME_COMMUNICATION=OFF` removes communication modules and the
serving executable; it does not disable backend, tokenizer, cache, diagnostics
or evaluation targets. Only serving applications link backend and transport.

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

CUDA and Metal may use different kernels, but they share the same model graph,
feature capability, sampling semantics, component names, and public behavior.
Architecture support is registered from components present in the model, not
from scattered per-model booleans. Optional Vision and MTP components default
to enabled when both the graph and weights provide them; missing weights or an
explicit user override disables them.

## Change checklist

Before submitting a runtime or quantization change:

1. Identify the owner: core, transport, scheduler, shared Engine/model,
   backend operation/storage, importer, or application.
2. Search all architectures and both backends for an equivalent implementation.
3. Extract reusable behavior before extending it; keep model adapters thin.
4. Test changed behavior and protect critical ownership rules under `tests/`.
   Source-boundary checks and compilation do not establish numerical accuracy
   or performance; use real model weights for those claims.
5. Run the focused native tests, the Python suite, and the full CTest suite when
   the change touches shared runtime behavior. For build-boundary changes,
   also build and test with `MFQ_BUILD_RUNTIME_COMMUNICATION=OFF`.
6. Do not publish generated files, local paths, credentials, model weights, or
   disposable benchmark artifacts.

Code review must reject architecture-local duplication even when it works for
the first model. The maintenance cost is paid by every model added afterward.
