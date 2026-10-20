# MFQ 已知 BUG 账本

本文件只保留未解决的项目问题，剩余编号保持稳定，不复用、不重新编号。

状态口径：

- **待修复**：现象明确，尚未处理。
- **结构债务**：所有权/边界问题，终态定义在 [refactor.md](refactor.md)，落地顺序与门槛在 [plan.md](plan.md)，本文件只登记状态与关闭条件，不复制计划。
- **验证缺口**：项目所需的测试或对照尚不完整。
- **待复核**：早期文档记录，本轮未重新验证，不断言现在仍复现；关闭前需按条目给的验证条件重跑。

## 状态一览

| 编号 | 类别 | 状态 | 摘要 |
| --- | --- | --- | --- |
| B11 | 真实缺陷 | 待修复 | float projection 融合残差在分段 up projection 下断言失败 |
| B14 | 验证缺口 | 待补 | 采不到尾延迟、logits 基线与随机采样分布 |
| B15 | 工具缺陷 | 待修复 | `mfq-cuda-include-graph-test` 被重复注册 |
| B16 | 真实缺陷 | 待复核 | Python 全量回归遗留失败、`hardware_identity()` 缓存污染与失效浅测试 |
| B17 | 真实缺陷 | 待复核 | `--check-dsv4-attention` 的 native Float8 转换错误 |
| B18 | 真实缺陷 | 待复核 | Metal depth controller 未跟进公共接口 |
| B19 | 真实缺陷 | 待复核 | 全量回归跳过项：MSVC 限定与旧 native 可执行路径 |
| B20 | 真实缺陷 | 待复核 | Studio 聊天页不显示 `conversation.error`，无清除入口 |
| B21 | 真实缺陷 | 待复核 | Studio 实时语音 `submitText()` 竞态，未发送成功就清空草稿 |
| B22 | 真实缺陷 | 待复核 | Studio 恢复完成前可切会话/模型，同步失败丢临时回答 |
| B23 | 真实缺陷 | 待复核 | Studio 评测进入终态后结果页不自动刷新 |
| B24 | 真实缺陷 | 待复核 | Studio `conversation.transitioning` 未纳入全局 busy |
| B25 | 真实缺陷 | 待复核 | Studio 删除会话/淘汰历史不清理 IndexedDB 语音 Blob |
| B26 | 结构债务 | 待处理（plan 阶段 6） | Duplex 走 `ControlPending/ControlCompletion` 第二生命周期 |
| B27 | 结构债务 | 待处理（plan 阶段 7） | 只排列回调的共享模板 |
| B28 | 结构债务 | 待处理（plan 阶段 2/4/7） | 模型继承与绑定链；Metal 未接入共享 model/engine |
| B29 | 结构债务 | 待处理（plan 阶段 2） | config/metadata/RuntimeComponents 重复状态需手动同步 |
| B30 | 结构债务 | 待处理（plan 阶段 6） | 视觉数学留在 CUDA engine 装配头 |
| B31 | 结构债务 | 待处理（plan 阶段 6） | MiniCPM 语言模型 move 后用 `dynamic_cast` 找回 |
| B32 | 结构债务 | 待处理（plan 阶段 3） | family 专用 batch 驱动尚未归入共享 Executor |
| B33 | 结构债务 | 待决（开关归属未定；未复现故障） | 调试开关归属未定；若改成缓存读取会破坏测试内 `setenv` A/B |
| B34 | 结构债务 | 待处理（plan R4，随阶段 2/3） | `CudaStores` 只是字段包装，Store 所有权未收口 |
| B35 | 真实缺陷 | 待修复 | MXFP8 反向的 FP16 GEMM 回退在 16 行输入下超过既有容差 |
| B36 | 真实缺陷 | 待修复 | shared-input mixed Gate 与既有路径不能逐字节一致，K2 基线同样复现 |

## 真实缺陷（待修复）

### B11：`mfq-float-projection-test` 是真实缺陷，融合残差遇到分段投影时失败

- 状态：待修复。
- 位置：[gated_residual_fused.cu](../csrc/backends/cuda/kernels/gated_residual_fused.cu)。
- 影响：up projection 分段时融合残差路径直接断言失败，而不是走已验证的分开计算；混合/分段量化载荷会在运行时终止。
- 修复：按 [refactor.md](refactor.md)「格式描述与实现选择分开」——不满足 fused kernel 的格式或布局契约时，由同一算子执行已验证的分开计算；或在加载/批次准备阶段明确拒绝该组合。属数值路径改动，必须在 GPU 上验证。
- 验证/关闭条件：该用例通过，或该组合在加载/准备时报明确不支持；两种结果都要附实测记录。

### B15：`mfq-cuda-include-graph-test` 被重复注册

- 状态：待修复（工具与记录口径）。
- 位置：[csrc/tests/CMakeLists.txt](../csrc/tests/CMakeLists.txt) 与 [csrc/backends/cuda/CMakeLists.txt](../csrc/backends/cuda/CMakeLists.txt) 都 `add_test(NAME mfq-cuda-include-graph-test ...)`。
- 影响：同一条 CUDA DAG 检查在一次 CTest 运行里执行两遍，注册数被当成检查覆盖数时会高估。
- 修复：只保留一处注册（`csrc/tests/CMakeLists.txt` 与 backend 二选一），并让文档里的测试计数说明注册来源。
- 验证/关闭条件：`ctest -N` 中该名字只出现一次，总数与唯一检查数一致；文档引用测试数量时注明枚举口径。

### B16：Python 全量回归遗留失败与缓存污染

- 状态：待复核。
- 问题：依赖版本约束、`hardware_identity()` 缓存污染和失效的源码字符串断言影响全量回归。
- 验证/关闭条件：跑一次完整 Python 回归，逐项列出剩余失败并归因；污染项在独立进程或显式清缓存下复现；浅测试替换为行为测试后仍通过。

### B17：`--check-dsv4-attention` 的 native Float8 转换错误

- 状态：待复核。
- 现象：诊断报 `native Float8 conversion requires a format-specific kernel`。
- 影响：完整 DSV4 attention 回归无法完成，阶段 3 的第二模型验收缺这一格。
- 验证/关闭条件：运行 `--check-dsv4-attention` 与 `--check-dsv4-hc` 并通过，覆盖对应模型的实际行为。

### B18：Metal 侧 depth controller 未跟进合并

- 状态：待复核。
- 位置：[mlx_mtp.cpp](../csrc/backends/metal/runtime/mlx_mtp.cpp) 与 [depth.h](../csrc/engine/requests/depth.h)。
- 问题：Metal 仍以 `GenerationPolicy<MlxDsparkDepthController>` 实例化已合并的非模板 `GenerationPolicy`，需同步实际接口。
- 验证/关闭条件：在 macOS 上按原生 Metal 构建并运行相应契约用例，记录命令。

### B19：全量回归跳过项——MSVC 限定与旧 native 可执行路径

- 状态：待复核。
- 现象：全量回归里因 MSVC 限定写法与指向旧 native 可执行路径的测试而跳过。
- 影响：跳过项被计为通过时会高估覆盖率。
- 验证/关闭条件：修正限定与可执行路径后重跑，不再因失效的测试配置跳过。

### B20：Studio 聊天页不显示 `conversation.error`

- 状态：待复核。
- 现象：历史、附件、工具、编辑、重生成、语音和模式切换失败时，聊天页没有统一错误展示，也没有清除入口。
- 影响：失败静默，用户不知道请求是否发出。
- 验证/关闭条件：上述失败路径都在聊天页可见且可清除，附手动复现步骤或前端测试。

### B21：Studio 实时语音 `submitText()` 竞态

- 状态：待复核。
- 现象：`submitText()` 未等 `session.created` 且 `socket.send()` 成功就返回；连接关闭或失败不 reject，草稿仍被清空并记为已发送。
- 影响：语音输入丢失且不可恢复。
- 验证/关闭条件：连接失败时草稿保留并报错；成功路径在 send 完成后才清空，附前端测试或复现步骤。

### B22：Studio 恢复完成前可切会话/模型

- 状态：待复核。
- 现象：恢复完成前允许会话与模型切换，同步失败后临时回答丢失。
- 影响：切换会丢内容。
- 验证/关闭条件：要么禁止切换，要么按会话保存待恢复生成状态并在恢复后回填。

### B23：Studio 评测终态后不自动刷新结果

- 状态：待复核。
- 现象：评测任务进入终态后 `EvaluationsPage` 不刷新。
- 影响：需要手动刷新才能看到结果，容易误判任务卡住。
- 验证/关闭条件：终态后结果自动出现，附一次实测记录。

### B24：Studio `conversation.transitioning` 未纳入全局 busy

- 状态：待复核。
- 现象：创建或删除会话期间仍可继续发送。
- 影响：并发操作同一会话导致状态错乱。
- 验证/关闭条件：transitioning 期间输入被禁用，附复现步骤。

### B25：Studio 删除会话/淘汰历史不清理 IndexedDB 语音 Blob

- 状态：待复核。
- 现象：删除会话或淘汰历史记录时，对应的语音 Blob 留在 IndexedDB。
- 影响：本地存储持续增长，且已删数据仍留在浏览器里。
- 验证/关闭条件：删除后 IndexedDB 中对应 Blob 计数为 0，附一次实测。

### B35：MXFP8 反向的 FP16 GEMM 回退超过既有数值容差

- 状态：待修复，2026-10-20 在 RTX 4090、CUDA 13.2、Torch 2.14.1+cu130 上实测复现。
- 位置：[mx_matmul_backward.cu](../csrc/backends/cuda/kernels/quant/mx_matmul_backward.cu) 的大 batch 回退调用 [packed_backward.cuh](../csrc/backends/cuda/kernels/quant/packed_backward.cuh)；该共享 GEMM 使用 `CUBLAS_COMPUTE_16F`。
- 复现：`.venv/bin/python -m pytest -q 'tests/test_format_completion_cuda.py::test_mx_packed_backward_and_autograd_match_dequant[16-_mxfp8]'`，退出码 1。固定 seed 1716，权重 `[11,128]`、梯度 `[16,11]`，在 `rtol=0.006, atol=0.03` 下有 2/2048 个元素超标；超标元素中最大绝对差约 0.1122。
- 对照：权重解码与独立参考的最大绝对差为 0；相同半精度输入交给 Torch matmul（`allow_fp16_accumulation=False`）时，没有元素超过相同容差。上述两个反向文件与 `e590bad9` 一致，格式迁移没有修改其数值实现；冷编译暴露的几何 helper 可见性缺陷单独修复为共用已有实现。
- 影响：普通 MXFP8 的输入梯度精度不足，影响工具侧反向/校准；MXFP4/FP8-SQ 和原生 SQ fixture 回归通过不构成本条通过。
- 验证/关闭条件：在原有容差下通过该固定用例，并验证共享回退的其他格式、dtype、batch 和 Graph 路径；精度选择须作为独立数值修复验证，不随目录迁移改变容差。

### B36：shared-input mixed Gate 不能通过逐字节比较

- 状态：待修复；K3/K4 迁移没有改动该数值实现或容差。
- 位置：[`verify_shared_input`](../csrc/backends/cuda/tests/kernels/moe_ffn_pipeline_test.cpp) 的 mixed Gate/Up 普通与 Graph 对照，以及共享输入量化/heterogeneous decode 算子。
- 复现：用 `PYTHONPATH=. .venv/bin/python csrc/tests/quant/make_cpu_quant_fixtures.py /tmp/mfq-k3-quant-fixtures --experts 3` 生成固定 seed `20261005` 的范围夹具，然后运行 `build/cuda/mfq-moe-ffn-pipeline-test /tmp/mfq-k3-quant-fixtures --shared-input-check`。
- 现象：index 777，actual `0.000947952`，expected `0.000947475`，逐字节比较失败。不能用放宽相等判据把迁移计为数值修复。
- 基线：从 `28476704` 导出原头文件、原 `ops/moe/moe.cpp` 与原测试函数，独立编译该数值 provider/probe，使用未改动的 native kernel 库和相同夹具；相同 index 和相同数值差异复现，exit 1。本轮定向 CPU 成本观测覆盖 4 格式、16 次 warmup 和 32 次 decode，独立于此失败通过。
- 关闭条件：在原逐字节判据下通过普通/Graph、多格式及变化输入的比较，定位量化或累积顺序差异；与架构迁移分开验证。

## 结构债务

这些条目的终态定义在 [refactor.md](refactor.md)，落地顺序与逐阶段门槛在 [plan.md](plan.md)；本节只登记缺陷本身与关闭条件。待复核条目需先复现，再按关闭条件处理。

### B26：Duplex 走第二条请求生命周期

- 状态：待处理，随 plan 阶段 6。
- 位置：`ControlPending`/`ControlCompletion` 在 [engine/include/engine.h](../csrc/engine/include/engine.h)；Scheduler 侧第二套状态在 [scheduler/src/scheduler.cpp](../csrc/scheduler/src/scheduler.cpp)（`pending_control`、`duplex_active`）；CUDA 侧在 [backends/cuda/engine/cuda_engine.cpp](../csrc/backends/cuda/engine/cuda_engine.cpp)（`duplex_active`、`duplex_pending`、`duplex_stop`、`duplex_sequence`）。
- 影响：duplex 生成绕过 `EngineRequest → admit → step → EngineEvent → outbox`，取消、deadline、背压、terminal 与失败处理各有一套实现，容易只在一侧修好。
- 关闭条件：session 初始化与停止可保留 typed control，但每次 duplex 生成走统一请求流；独占执行由 Engine capacity/兼容性表达，不再由 Scheduler 与 CUDA Engine 分别维护第二套状态。

### B27：只排列回调的共享模板

- 状态：待处理，随 plan 阶段 7 的旧主链删除。
- 位置：[`attention_layer`](../csrc/engine/model/common/transformer_layer.h)、[`full_attention`](../csrc/engine/model/common/attention.h)（接收八个操作回调）、[`full_block.cpp`](../csrc/backends/cuda/models/common/full_block.cpp)；`gated_mlp` 有类似拆分。
- 影响：实际计算、融合条件与临时状态仍在后端，共享模板只排列 callback，读不出模型数学。
- 关闭条件：逐项判断删除模板后复杂度是消失还是转移；纯转发合回拥有实际行为的模块，保留有独立语义、验证或真实复用收益的组合。

### B28：模型继承与绑定链，Metal 未接入共享 model/engine

- 状态：待处理，随 plan 阶段 2（CUDA）与阶段 4（Metal）。
- 位置：入口 [`engine/model/common/causal_model.h`](../csrc/engine/model/common/causal_model.h)、[`backends/cuda/models/common/causal_model_ops.h`](../csrc/backends/cuda/models/common/causal_model_ops.h)、[`backends/cuda/models/qwen35/ops.h`](../csrc/backends/cuda/models/qwen35/ops.h)；链路 `CausalLm → CausalModelBase → CudaCausalOps → Qwen35Model → CausalResources`，另有 `ForwardOps` 与 `CudaSessionCodec`。Metal 的 `MlxEngine` 实现公共 `Engine` 但有自己的 generation，不使用 `EngineInstance` 或共享模型模板。
- 影响：多层直接依赖同一对象的 `config`、`metadata`、`blocks`、`execution`；实现者必须了解内部成员。共享性只由头文件不依赖后端来证明，实际是两套执行。
- 关闭条件：CUDA 侧沿 Qwen35 调用链确认没有 CausalModelBase/CudaCausalOps/ForwardOps；Metal 侧接入同一 Engine 装配与 family forward，不转发到旧 MLX generation loop。

### B29：重复状态需要手动同步

- 状态：待处理，随 plan 阶段 2。
- 位置：[`engine/model/qwen35/causal_lm.h`](../csrc/engine/model/qwen35/causal_lm.h) 的 `load_config` 逐字段复制到 `CausalLmMetadata`，`set_max_position_embeddings()` 同时改 config 与 metadata；[`backends/cuda/engine/model_components.h`](../csrc/backends/cuda/engine/model_components.h) 同时保存 `mtp_available` 等声明与实际状态。
- 影响：漏同步会产生互相矛盾的可变事实。
- 关闭条件：配置与加载结果各只有一份所有者，可派生状态不再单独存储；保留「模型声明 / 后端支持 / 权重实际加载成功」三者的语义区别。

### B30：视觉数学留在 CUDA engine 装配头

- 状态：待处理，随 plan 阶段 6。
- 位置：[`backends/cuda/engine/model_components.h`](../csrc/backends/cuda/engine/model_components.h) 前约 500 行包含视觉权重加载、LayerNorm、RoPE、编码、prompt 准备与单图缓存，最后才定义运行时组件。
- 影响：engine 装配头承担模型数学与缓存策略，改动面被放大。
- 关闭条件：按已有模型、算子与存储职责归位，engine 只做选择、装配与生命周期；迁移保留几何校验、缓存身份与分步执行行为。

### B31：MiniCPM 语言模型所有权靠 `dynamic_cast` 找回

- 状态：待处理，随 plan 阶段 6。
- 位置：[`backends/cuda/engine/model_components.cpp`](../csrc/backends/cuda/engine/model_components.cpp) 的 `load_runtime_components` 将语言模型 move 进 composite；[`backends/cuda/engine/cuda_engine.cpp`](../csrc/backends/cuda/engine/cuda_engine.cpp) `bind_language` 经接口再 `dynamic_cast<Model&>` 取回具体模型。
- 影响：moved-from 占位 + 类型擦除后再恢复具体类型，装配路径难验证。
- 关闭条件：模型包直接拥有 text/vision/audio/TTS 子模型并共享权重句柄，保留多模态与 duplex 资源的正确生命周期。

### B32：family 专用 batch 驱动尚未归入共享 Executor

- 状态：待处理，随 plan 阶段 3。
- 位置：[`BoundState`](../csrc/backends/cuda/engine/cuda_engine.cpp)、[`CudaBatchExecution`](../csrc/backends/cuda/engine/batch_execution.h) 与 [`Qwen4BatchExecution`](../csrc/backends/cuda/engine/qwen4_batch_execution.h)。family policy 与 `batch_executor_t<Model>` 选择物理实现，Qwen35 请求状态仍有专用绑定。
- 影响：增加第二个绑定不等于共享执行链已落地；family 专用 batch 驱动与旧 `ContinuousBatch` 仍需迁入共同的计划、执行和事务规则。
- 关闭条件：第二种数学/状态结构通过同一公共 Planner/Executor/Store，通用规则不新增 family 分支，并通过对应真实模型与取消、恢复、预算验证。

### B33：native tensor/kernel 调试开关归属未定，改成缓存读取会破坏进程内切换

- 状态：待决，开关归属与读取语义需要明确。
- 影响：开关归属未定使 Engine policy 与底层调试开关混在一起；真正的风险在改动兼容性：先改读取方式而后决定归属，会静默打断依赖 `setenv` 的测试。
- 关闭条件：先明确开关归属（哪些进 `CudaExecutionConfig`，哪些只作为进程级调试/graph dump 开关留在底层），再决定读取方式；若改为缓存，依赖进程内 `setenv` 切换的 A/B 测试仍通过。

### B34：`CudaStores` 只是字段包装，Store 所有权未收口

- 状态：待处理，即 plan §1.4 的 R4；实际 Store 随阶段 2 的 Qwen 与阶段 3 的 DeepSeek 落地，不能提前标完成。
- 位置：[`backends/cuda/execution/cuda_execution.h`](../csrc/backends/cuda/execution/cuda_execution.h) 中 `CudaStores` 持有 `kv_memory`、`moe_expert_cache` 和 `qsa_kv_store`，`CudaStores stores;` 仍由 `CudaExecutionContext` 持有。
- 影响：没有公共 StateStore/WeightStore，状态修改权限与回收条件没有单一所有者。
- 关闭条件：见 plan §1.4 R4 的验收——从加载、执行到取消/reload 都能定位唯一管理者；延迟 I/O/设备完成时仍被使用的页、slot、权重和任务资源不能提前复用；失败预留可回收、过期 generation 不能提交。仅把成员移进另一结构体不算完成。

## 验证缺口

### B14：验收工具采不到尾延迟、logits 基线与随机采样分布

- 状态：待补。
- 位置：原生 [diagnostics](../csrc/backends/cuda/commands/diagnostics.cpp) 的 trace、生成与基准入口。
- 影响：验收契约要求的「随机采样接受/拒绝与替换分布」和「尾延迟」没有基线，阶段 5 的 MTP stochastic 验收没有对照物。
- 剩余工作：复用 `--dump-block-trace` 采 logits，补并发延迟与随机采样分布的采集和对照，记录命令、退出码与产物。
- 验证/关闭条件：三项都有可复现命令与数值，不再标未采集；未测和工具退出成功均不能代替数值验收。
