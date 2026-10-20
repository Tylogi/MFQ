# MFQ 公共推理架构重构实施计划

本文只保留剩余迁移工作和项目级验收条件。目标架构见 [refactor.md](refactor.md)，未解决问题见 [bugs.md](bugs.md)，当前代码结构见 [architecture.md](architecture.md)。完成项从计划删除。

目标是让不同后端复用同一份模型组合、执行流程和状态语义；不为旧 C++ 内部接口保留兼容层，不借重构改变算术、精度或容差。

## 0. 剩余工作简表

| 阶段 | 剩余工作 |
| --- | --- |
| 0 验证入口 | 补模型 logits、chunked prefill、随机采样分布与并发延迟的对照和回归 |
| 1 公共骨架 | 完成 K1/K2 剩余依赖迁移，明确 Engine、Store 与 Executor 的装配和所有权 |
| 2 首条共享模型链 | 实现 Qwen35 hybrid eager 路径，替换旧继承链与双层 batch 驱动 |
| 3 第二模型与动态资源 | 验证第二种数学/状态结构，接入专家驻留、offload 与动态 I/O，完成 core 拆除 |
| 4 第二后端 | 将 Metal 接入共享 Engine 和 family forward，再稳定公共接口 |
| 5 状态事务 | 统一 session/prefix、MTP、回滚与输出提交 |
| 6 优化与消费者 | 接入 Graph、并行、多模态和 Duplex，逐模型迁移实际调用方 |
| 7 删除旧链 | 删除旧模型绑定、重复生成循环与转发层 |

## 1. 目录与依赖迁移

### 1.1 代码落点

终态目录与逐文件归属以 [refactor.md](refactor.md) 为准，剩余迁移包括：

- 将 `core/` 的格式、模型读取和文件 I/O 拆入 `formats/`、`model_source/`、`components/io/`。
- 将共享配置和 forward 收敛到 `engine/model/<family>/`，删除旧 Ops/继承/ForwardOps 绑定链。
- 将状态、权重与执行规则分别落实到 `engine/state/`、`engine/weights/` 和 `engine/executor/`；后端保留物理存储与数值执行。
- 将 CUDA `engine/` 收敛为装配入口，迁走其中的生成、事务和可复用模型行为。

复用现有请求 map、`plan_batch`、页分配与 slots/ranker，不为新名称增加包装类。Transport、tokenizer、CLI 与诊断保持各自职责。

### 1.2 依赖检查

迁移时同步 `test_layer_independence.py`、`test_cuda_include_graph.py` 和 `test_engine_include_graph.py`，继续覆盖真实消费者及正反例：

- 公共层不依赖具体后端；通用请求、Planner 和 Store 不按 family 分支。
- ops 不依赖模型文件工厂、Store 策略或生成驱动。
- Store 不依赖 Model/Executor，Model 不依赖 Executor。
- 新实现不 include 旧绑定链；文件与 target 图无循环。
- 例外按具体消费者列出，迁完即删除，不通过排除整个目录规避检查。

### 1.3 CMake 与规范同步

公共非模板实现由 `mfq-engine` 编译，模板由实际后端和测试实例化；公共库不链接设备库，也不公开全部 implementation 目录。

每批迁移同步 CUDA、Metal、CPU、工具绑定、bench 与测试的实际消费者，以及 [CONTRIBUTING.md](../CONTRIBUTING.md)、[运行时说明](../csrc/README.md) 和本文。旧路径直接更新，不保留转发头或聚合库。

### 1.4 Store 所有权（R4）

`CudaStores` 仍由执行上下文持有，尚未建立目标 StateStore/WeightStore 的完整行为：

- Engine 唯一持有 Store，Store 组合拥有原生池和载荷；Executor 与模型取得引用、view 或 lease。
- 身份、预算预留、版本、独占写、提交/恢复和回收由对应 Store 管理。
- 先落实 Qwen 的 KV/GDN/卷积状态与常驻权重，再迁移 MoE 专家缓存和 QSA 存储。
- 同步加载、reset/reload、MTP、Graph 和诊断消费者；最后一个旧消费者迁完后删除 `CudaStores`。

退出条件：失败预留可回收，过期 generation 不可提交，仍被 I/O 或设备工作引用的页、slot、权重和任务不能提前复用。字段搬家不算所有权迁移完成。

### 1.5 推理与工具构建隔离

纯 Engine 改动只构建推理及受影响的原生测试；共享 kernel、格式、路径或绑定变化时验证实际工具消费者。保留既有数值失败，不借迁移修改容差。

#### G3：共享 kernel 的绑定边界

继续消除共享 `.cu` 对 Torch/pybind 与双模式 `backend.h` 的传递依赖。复用 view、裸指针和 stream 入口，Tensor 校验、分配和 stream 取得留在宿主 adapter；各消费者的数值编译设置保持不变。

#### G4：按改动范围验收

原生推理、诊断/评估和工具扩展保持独立构建。共享源码变化同时验证原生与实际工具调用；依赖图和源码清单检查不能代替数值行为测试。

### 1.6 拆除 core：K1–K4

| 步骤 | 剩余工作 |
| --- | --- |
| K1 底层依赖 | 迁移 formats、model_source、剩余文件 I/O 与 model-graph 中的视觉算法，清除 CUDA kernels 的旧 core 和运行策略依赖 |
| K2 契约与模型 | 合并请求/采样值契约，归位 StepSequence、grid vision 与公共头，收口 ModelSource 的 compat 暴露 |
| K3 状态与放置 | 将 prefix/QSA、在线 CPU 成本测量、MoE 放置和分片规则接入 Store/Executor |
| K4 删除入口 | 删除 `csrc/core/`、`mfq-core`/`mfq::core`、`MFQ_CORE_INCLUDE_DIR` 及全部活动消费者 |

每批更新实际 include/target 闭包、扩展清单和生成文件路径。公共能力不复制进各后端，也不整体改名成另一个聚合目录。core 删除不等于 Store 或跨后端迁移完成。

## 2. 实施阶段

### 阶段 0：补齐验证入口

补模型 logits、hybrid chunked prefill、随机采样分布、并发延迟和资源生命周期的对照。采集与比较工具需有明确判据，导出成功或同 seed 可复现不能代替数值与分布验证。

### 阶段 1：公共骨架与构建关系

完成 K1 与独立的 K2 迁移。用真实调用明确输入、状态要求、lease 和 completion，避免空接口；保持公共 target 无设备链接依赖。Store 行为随阶段 2/3 的实际路径落实。

### 阶段 2：共享 Qwen35 hybrid eager 链

共享 forward 直接组合完整 full attention、GDN 和卷积算子；实现 StateStore/WeightStore、Preparation/InFlightBatch 与 Executor。新链不得再通过 `CausalModelBase`、`CudaCausalOps` 或 `ForwardOps` 转回旧模型。

验证单请求与 batch、chunked prefill、取消、背压、延迟 completion 和状态恢复。一次 forward 完成不代表事务提交或资源回收完成。

### 阶段 3：第二模型与动态资源

接入 DeepSeek 等第二种数学/状态结构，覆盖动态专家、QSA、offload 与 I/O；通用 Planner/Executor/Store 不新增 family 分支。完成 K3/K4 与 R4 的动态资源管理。

### 阶段 4：Metal 验证公共设计

Metal 使用同一份 Engine 执行流程和 family forward，替换独立生成和模型实现。以真实第二后端检验公共语义，再稳定接口；共享接口本身不等于共享行为。

### 阶段 5：公共 session/prefix/MTP 与输出事务

统一 target、predictor、KV、recurrent、卷积、位置和 RNG 的提交/回滚。覆盖 session restore/fork、零/部分/全部接受、拒绝与 replacement，以及提交失败后的恢复；确认提交后才发布输出。

### 阶段 6：优化路径、消费者与逐模型切换

将 Graph、并行、offload 和多模态接入共同流程，保持 eager 与优化路径语义一致。MiniCPM 模型包直接持有子模型，Duplex 生成迁入统一请求流，删除 move 后 `dynamic_cast` 找回语言模型和独立生成生命周期。

### 阶段 7：删除旧主链

按真实消费者删除旧继承/绑定、双层 batch 驱动、重复 generation/MTP/session 实现和纯转发层。删除前验证各模型已有能力，不能以目录迁移代替行为迁移。

## 3. 工厂切换与删除门槛

- 同步加载、generation、MTP、session 和 KL 的实例化入口；`MFQ_CUDA_EACH_MODEL` 不是新旧执行链开关。
- 保留一种明确的执行链选择，不能增加模型内部隐式 fallback。
- 旧消费者归零、受影响行为验证通过后再删除旧实现，不保留永久兼容 facade。

## 4. 全阶段测试与验收

### 4.1 按实际影响选范围

覆盖受影响的公共行为、后端实现和实际消费者。结构检查、fixture、数值与模型回归分别说明证据范围；未测不能计为通过。

### 4.2 必须覆盖的行为

- 请求准入、背压、取消、deadline、恰好一个 terminal，以及资源清理。
- 权重加载、格式校验、损坏/截断输入与混合格式；普通/融合和 eager/Graph 的数值一致性。
- 状态 save/restore/fork、MTP rollback、失败恢复和输出提交。
- 延迟 I/O、最后消费者、lease、slot generation、预算与 reload/shutdown。
- 模型实际支持的 batching、并行、多模态和 Duplex 能力。

### 4.3 触发与阶段退出

依赖变化更新图检查与反例；行为变化运行对应原生和消费者回归。第二模型、第二后端及旧链删除分别验证对应能力，不能用另一条路径的历史成功代替。

### 4.4 验收记录

记录源码与产物身份、固定输入、参数、实际命令和结果。数值容差事先确定；skip、缺参分支、工具退出成功和导出完成均不单独构成通过。

### 4.5 失败与性能判据

既有失败与新增失败分别归因。性能结论使用固定模型、输入、采样和预算，分开记录延迟、吞吐与资源开销，不通过更换基线掩盖退化。

### 4.6 回退

回退前停止准入，drain/cancel 活动工作并完成清理；lease、snapshot 与执行图不跨新旧实现迁移。回退后重新验证实际影响范围。
