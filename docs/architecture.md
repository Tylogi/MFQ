# MFQ 当前项目架构

本文只描述**现在实际存在**的架构：模块、数据流、目录、控制流、接口字段和资源持有。目标设计与「必须/禁止」规则在 [refactor.md](refactor.md) 与 [CONTRIBUTING.md](../CONTRIBUTING.md)；实施顺序与验收条件在 [plan.md](plan.md)；已知缺陷在 [bugs.md](bugs.md)。描述现状不等于要求未来继续走旧链。

## 1. 顶层模块与数据流

四个层次：MFQStudio 前端、Python server/CLI、离线校准与量化、native C++ runtime。

```text
MFQStudio（TypeScript SPA，MFQStudio）
   │ HTTP / WebSocket
   ▼
Python server（mfq/server）
   │ 进程管理 + HTTP 客户端
   ▼
native worker 子进程（mfq-runtime --transport http --host 127.0.0.1 --port N）
   │ Transport → Scheduler → Engine → backend ops / kernels
   ▼
CUDA 或 Metal

离线：mfq/commands → mfq/calibration + mfq/quantize + mfq/formats + mfq/architectures
      → MFQ 产物 → native model_source + storage 读取
```

下文源码路径以仓库根目录为基准。

### Python 侧（`mfq`）

| 目录 | 当前职责 |
| --- | --- |
| `commands/` | CLI 入口：`build.py`、`serve.py`、`quantize.py`、`alphaq.py`、`optimize.py`、`solve_ew.py` |
| `server/` | `api/`（OpenAI/Anthropic 兼容路由、auth、`static.py` 的 `SPAStaticFiles`）、`protocol/`、`services/`（jobs、hardware、benchmark、tool_jobs 等）、`state/`（模型 catalog）、`vision/` |
| `server/runtime/` | 与 native worker 的边界：`native.py` 的 `NativeRuntime` 用 `subprocess.Popen` 启动本地 HTTP worker（`--transport http --host 127.0.0.1 --port <free>`，`base_url = http://127.0.0.1:<port>`）；`client.py` 是基于 httpx 的 transport-neutral 客户端；`backend.py` 是 Python API 适配层；`runtime_pool.py` 管理 runtime 进程并按模型路由；另有 `capabilities.py`、`cluster.py`、`hf_tokenizer.py`、`host_memory.py`、`prefix_cache_text.py`、`qsa_kv_policy.py` |
| `calibration/`、`quantize/`、`formats/`、`architectures/` | 离线数据统计、精度搜索、权重编码、产物 schema 与 HF 配置读取 |
| `kernels/cuda/` | Python 侧只有 torch 绑定与清单：`mfq_cuda.cpp`、`_ext.py` 和各算子 wrapper `.py`。共享 `.cu/.cuh` 已全部位于 `csrc/backends/cuda/kernels`，Python 扩展从这些原生源码里取子集编译 |
| `runtime/` | 仍存在参考与工具实现（`mlx_*.py`、`causal_lm.py`、`model.py`、`minicpmo45*.py`、`torch_linear.py`、`flash_next_worker.py`）。它们不是生产推理入口；「生产推理不走 Python」是 [PYTHON_SIMPLIFICATION.md](../../docs/PYTHON_SIMPLIFICATION.md) 的规则，不是已完成的删除记录 |

MFQStudio 由 `mfq/server/api/static.py` 提供静态资源，走 `api/` 路由与 realtime WebSocket。

## 2. native runtime 的控制流（当前）

`Runtime = Transport + Scheduler + Engine` 是装配与生命周期职责的划分。没有单独的顶层 `csrc/runtime/` 目录，也没有名为 `Runtime` 的装配类；装配发生在可执行程序的 `apps/`、`commands/` 里。（`csrc/backends/metal/runtime/` 确实存在，那是 Metal 的执行实现目录，见 §4。）

```text
transport threads (stdio / HTTP)
  -> MfqScheduler mailbox_ (std::deque<Command>)
  -> scheduler loop -> Engine::step(eligible) -> EngineStepResult
  -> MfqScheduledRequest outbox -> transport
```

| 组件 | 位置 | 当前事实 |
| --- | --- | --- |
| Transport | `csrc/transport/`（`stdio.cpp`、`http.cpp`、`protocol.cpp`、`common.*`；接口 `include/transport.h` 的 `MfqTransport`） | 仅在 `MFQ_BUILD_RUNTIME_COMMUNICATION=ON` 时编译 |
| Scheduler | `csrc/scheduler/`（`include/scheduler.h`、`src/scheduler.cpp`） | `MfqScheduler` 单 loop 线程；`Limits` 约束请求数、输入字节数和输出队列的事件数/字节数；`MfqScheduledRequest::wait()/done()` 是 outbox 读取接口 |
| Engine 接口 | `csrc/engine/include/engine.h` | 唯一跨后端多态边界 |
| 公共 Engine 生命周期 | `csrc/engine/src/instance.h` 的 `EngineInstance<Backend>` | CUDA 走这条：`load_cuda_engine()` 返回 `std::make_unique<EngineInstance<CudaBackend>>(...)`（`backends/cuda/engine/cuda_engine.cpp`） |
| Metal Engine | `csrc/backends/metal/runtime/mlx_engine.h` | `MlxEngine final : public mfq::engine::Engine`，由 `apps/mfq_decode_mlx.cpp` 装配；不使用 `EngineInstance` |

### Engine 的实际 Interface（`csrc/engine/include/engine.h`）

```cpp
EngineInfo info() const;
Admission admit(EngineRequest&& request);   // Admission { accepted, deferred }
void cancel(const RequestId& id);
EngineStepResult step(const std::vector<RequestId>& eligible);
EngineStatus status() const;                // { available, healthy }
SessionResult session(const SessionCommand& command);
std::int64_t reload(std::int64_t context);
void shutdown();
ControlResult control(ControlRequest request);
```

- `admit` 只有 `accepted` / `deferred` 两个结果；没有第三种 typed rejection variant，错误经异常或由调用方处理。
- `EngineRequest` 字段：`id`、`InferenceInput input`、`optional<deadline>`、`optional<DiagnosticTokenInput>`。
- `EngineStepResult` 字段就是 `advanced`、`events`、`status`、`has_work`、`wake_at`、`control`、`control_advanced`、`has_control_work`。没有独立的 blocked / released / terminal ID 列表；blocked 与 terminal 由 events 加 Scheduler 自己的逻辑状态得出。
- 事件是 `EventData = variant<PrefillProgress, OutputDelta, UsageUpdate, Completed, Cancelled, Failed>`。`OutputDelta` 实际是 `{ token_ids, diffs }`（`OutputDiff` 列表），不是 text / reasoning / tool_delta 之类的示意字段。`terminal()` 只认 `Completed`、`Cancelled`、`Failed`；拒绝不是额外的 terminal variant。
- 控制路径当前存在：`ControlRequest`（`DecodeTokens`、`PrepareDuplex`、`PrepareDuplexStep`、`MfqDuplexSessionParams`、`MfqDuplexStepInput`、`StopDuplex`、`RuntimeMetrics`、`ScoreText`、`ScoreTokens`）返回 `ControlResult`；其中 `ControlPending` 表示由后续 `step()` 发布恰好一个 `ControlCompletion`。

## 3. 当前模型组织与资源持有

- 共享 family 定义在 `csrc/models/<family>/`（`config.h/.cpp` + `causal_lm.h`）。`models/common/causal_model.h` 提供 `CausalModelBase<Backend, Derived>`；`CausalLm<Backend>` 由各 family 自己的 `causal_lm.h` 定义（例如 `models/qwen35/causal_lm.h`）。
- CUDA 绑定在 `csrc/backends/cuda/models/`：`common/causal_model_ops.h` 的 `CudaCausalOps<Model> : Model, CudaCausalModel`与 `attention_ops.h`；`qwen35/ops.h` 实例化 `CausalLm<CudaCausalOps<Qwen35Model>>`。另有 `ForwardOps` 与 `CudaSessionCodec`。
- **这套共享目前只覆盖 CUDA 侧**：Metal 有自己的 family 实现（`backends/metal/models/` 下的 `mlx_*`），不实例化这些模板，也不接 `EngineInstance`。见 [bugs.md B28](bugs.md)。
- Engine 模块目录：`csrc/engine/{include,src,requests,batch,state,weights,executor,text}`。`include/` 是 PUBLIC，其余模块目录在 `engine/CMakeLists.txt` 里是 PRIVATE，只有 `backends/` 与 `tests/` 把它们加进 include path。
- 资源持有：`CudaStores` 持有 `kv_memory`、`moe_expert_cache` 和 `qsa_kv_store`，由 `CudaExecutionContext` 持有（`ops/execution/cuda_execution.h`）；`CudaEngineState` 创建 family 的物理 batch storage，`CudaBatchExecution` 按引用借用。公共 StateStore/WeightStore 尚未拥有这些资源（[B34](bugs.md)）。
- 连续批处理：共享 `ContinuousBatch` 在 `engine/executor/driver.h`，由 `backends/cuda/engine/batch_execution.*` 驱动；当前由 family policy 与 `batch_executor_t<Model>` 选择实现，包含 Qwen35 与单独的 Qwen4/Flash-Next batch 绑定；Qwen35 请求状态仍专用绑定，尚未收敛到目标共享 Executor（[B32](bugs.md)）。
- host parallel、拓扑、亲和性和 host copy 位于 `backends/cpu/native/`，专家调度位于 `backends/cpu/execution/`；CUDA native 的线程原语依赖不引入 CPU 专家调度。
- `csrc/core/` 仍存在，是 `mfq-core` INTERFACE 聚合 target（`csrc/core/CMakeLists.txt`），里面混有模型文件读取、格式解码、模型几何、请求值契约、KV 磁盘存储与专家放置策略。

## 4. CUDA 与 Metal 的实际差别

| 方面 | CUDA | Metal |
| --- | --- | --- |
| Engine 装配 | `EngineInstance<CudaBackend>` | `MlxEngine` 自己实现 `Engine` |
| family forward | `csrc/models/<family>` + `CudaCausalOps` 绑定 | `backends/metal/models/` 独立实现，按 backbone 在 `mfq_decode_mlx.cpp` 分派 |
| 交错 batch | `ContinuousBatch` + `batch_execution.*` | `MlxEngineModel` 默认不支持 batch，`execute()` 为空实现；`MlxEngine` 委托该模型接口 |
| CUDA Graph / DecodeGraph | `backends/cuda/engine/decode_graph.*` | 整个 `backends/metal` 无 `DecodeGraph` 引用 |
| MTP | `engine/requests/{depth,speculate}.h` + `backends/cuda/engine/mtp.cpp` | `runtime/mlx_mtp.*`；`mlx_mtp.cpp` 仍引用 `GenerationPolicy<MlxDsparkDepthController>` |
| session / prefix | `engine/state` 规则 + `storage/session` | `mlx_paged_session_codec` |
| 构建门槛 | `MFQ_BUILD_CUDA_RUNTIME` 决定是否配置 CUDA；关闭后公共库无需 CUDA SDK | `MFQ_BUILD_METAL_RUNTIME` 仅 Apple，且需要 `MFQ_MLX_ROOT` 下的 MLX 产物 |

## 5. 构建边界（当前）

`csrc/CMakeLists.txt` 的配置顺序：`components/json` → `core` → `backends/cpu` → `models` → `components/ggml` → `components/tokenizer` →（`components/http`）→ `engine` → `scheduler` →（`transport`）→（`backends/metal`，Apple）→（`backends/cuda`，非 Apple）→（`tests`，`BUILD_TESTING`）。CUDA/Metal 子目录各由 `MFQ_BUILD_CUDA_RUNTIME` 与 `MFQ_BUILD_METAL_RUNTIME` 控制；关闭 CUDA 时不启用 CUDA 语言、不查找其 SDK、不生成设备 target。HIP 也有独立后端开关，CUDA 与 HIP 不能在同一构建目录同时启用。

产物：CUDA 侧 `mfq-runtime`、`mfq-diagnostics`、`mfq-eval`（可选 `mfq-runtime-torch`，需要通信开启）；Metal 侧 `mfq-decode-metal`、`mfq-perplexity`。`mfq/commands/build.py` 默认只请求当前后端的推理 target（CUDA `mfq-runtime`，Metal `mfq-decode-metal`），diagnostics 与 eval 需要显式 `cmake --build ... --target`；服务只用推理程序，缺工具时在任务里报错。当前构建入口和验证范围见 [plan.md](plan.md) §1.5。

`MFQ_BUILD_RUNTIME_COMMUNICATION=OFF` 去掉通信模块与 `mfq-runtime`，保留 backend、tokenizer、prefix cache、diagnostics 与 eval。

## 6. 过渡限制（当前事实）

- Duplex 走第二条控制路径：`ControlPending` / `ControlCompletion`（`engine/include/engine.h`）、Scheduler 侧 `pending_control` / `duplex_active`（`scheduler/src/scheduler.cpp`）、CUDA 侧 `duplex_active` / `duplex_pending` / `duplex_stop` / `duplex_sequence`（`cuda_engine.cpp`）。见 [B26](bugs.md)。
- Metal 保留独立 generation；`mlx_mtp.cpp` 的 `GenerationPolicy<MlxDsparkDepthController>` 引用残留源码可见，接口仍需同步。见 [B18](bugs.md)。
- Store 所有权未收口（`CudaStores`）。见 [B34](bugs.md) 与 plan §1.4 R4。
- 没有 `csrc/engine/model/`，没有 `Preparation` / `InFlightBatch`；旧继承链与双层 batch 驱动仍是主链。见 plan 阶段 2。
- `core/` 聚合仍在。K1 的 host/CPU 原语、专家调度与文件 I/O 已拆出；纯格式头、冻结码表与 NINT/NVQ/MFE 范围读取归 `formats`（头 target `mfq-formats` 与编译 target `mfq-quant-readers` 分开），CUDA kernel 不再取得 core 的 include 路径或链接 `mfq-core`。native 的分配失败异常仍私有依赖 core，随 K2 迁移。格式 schema、ModelSource、旧标识输入边界、视觉算法与 K2–K4 仍待迁移。见 plan §1.6。
- 已知非绿：`mfq-float-projection-test`（[B11](bugs.md)）、`mfq-cuda-include-graph-test` 重复注册（[B15](bugs.md)）、普通 MXFP8 的 16 行反向回退误差（[B35](bugs.md)）。

## 7. 规则与目标在哪里

- 目标架构、目录终态、所有权与依赖方向：[refactor.md](refactor.md)。
- 当前必须遵守的工程规则与改动清单：[CONTRIBUTING.md](../CONTRIBUTING.md)。
- 实施顺序、逐阶段退出条件和验收要求：[plan.md](plan.md)。
- 缺陷与验收缺口：[bugs.md](bugs.md)。
