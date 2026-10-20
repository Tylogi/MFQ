# MFQ C++ 服务端目标架构

本文记录 C++ 服务端重写的目标设计，实施顺序见 [迁移计划](backend-plan.md)。
这是目标架构，不表示已经完整实现。当前产品仍使用 `mfq/server/` 的 Python 服务和
`cpp_runtime/` 下的原生推理代码；`cpp_runtime/server/` 已提供独立原生服务入口、
鉴权、SQLite 基础设施和本地模型目录。阶段一基础设施已通过 Linux 验收，
Windows/macOS 尚待验证；阶段二已实现目录部分，模型加载、同进程推理和其余业务接口尚未接入。
目标目录统一命名为 `csrc/`，承接 `refactor/cuda` 的目录调整。

## 1. 已确认的设计

- 本地服务与生产推理运行在同一个 C++ 进程中，直接调用原生接口。
- 移除 Python 服务与原生 runtime 之间的本地 HTTP、stdio 协议及进程管理。
- 保留 Scheduler 与 Engine 的职责分工，本次迁移不要求合并它们。
- 应用管理每个模型实例的 Engine 和 Scheduler，统一构造、停止和销毁。
- Scheduler 与 Engine 的请求记录通过 `RequestId` 关联，不增加第三张桥接表。
- 公共 HTTP、SSE、WebSocket 和远端节点通信继续保留。
- 服务、Studio 和发布包的产品功能最终不依赖 Python，包括后台调用的工具。

独立研究、训练和离线实验脚本是否全部迁到 C++，尚未确定。它们的去留不能作为
提前删除产品功能的理由，也不能成为产品运行时的隐式 Python 依赖。

## 2. 进程、实例与调用链

```text
Studio / 浏览器 / 外部客户端
        │ HTTP / SSE / WebSocket
        ▼
C++ 服务进程
  server/api
        │ 协议校验、鉴权、业务操作
        ▼
  server/models：查找已加载实例
        │ 直接调用
        ├── 模型实例 A：Scheduler → Engine → 模型 / 算子 / kernels
        └── 模型实例 B：Scheduler → Engine → 模型 / 算子 / kernels
```

当前 MFQ 中 Scheduler 和 Engine 已经在同一个原生 runtime 进程内。
本次合并的是 Python 服务进程与原生 runtime 进程，不是合并 Scheduler 与 Engine
两个进程；它们原本就不存在这层进程边界。

默认一个 Engine 对应一个模型包，可以使用一组设备。不同模型包使用不同实例，
不混入同一物理批次。应用创建 Engine，再为它创建 Scheduler；停止时先关闭入口，
再停止准入、取消或排空请求，最后释放执行与模型资源。

同一实例的活动 Engine 操作由 Scheduler 的循环调用。HTTP 线程提交命令和读取
有界结果队列，不直接并发调用 `Engine::step()`、reload 或缓存修改接口。
设备执行、CPU 预处理和网络 I/O 可以有各自的线程，不额外建立第二套请求驱动循环。

Runtime 在这里表示应用的装配与生命周期职责，不新增一个同义的顶层 `runtime/`
目录或转发类。也不新增 `workers/` 或 `inference/` 作为逐请求转发层。

## 3. 目标目录

```text
csrc/
  CMakeLists.txt
  apps/
    server_main.cpp         CLI、配置、后端选择与应用装配
  server/
    server.h / server.cpp   服务启动、停止与模块装配
    api/
      http.cpp              监听、路由注册、中间件
      openai.cpp            OpenAI 协议适配
      anthropic.cpp         Anthropic 协议适配
      routes/               Studio 业务接口
      events.cpp            SSE / WebSocket 事件编码
      static.cpp            Web UI 静态资源
      openapi.json          公共接口契约
    auth/                   密钥、权限与鉴权策略
    models/                 模型目录、运行配置与已加载实例
    sessions/               对话、消息分支、回复记录与生成预设
    jobs/                   后台任务状态、进度、取消与恢复
    hub/                    模型仓库查询、下载与续传
    media/                  上传、媒体资源与文档提取
    evaluations/            数据集、评测配置、结果与对比
    cluster/                远端节点管理与调用
    storage/                SQLite、事务与数据库迁移
  scheduler/                请求准入、调度、取消与输出背压
  engine/                   共享推理执行、模型、权重与状态管理
  formats/                  模型产物与量化格式
  model_source/             HF / GGUF / MFQ 读取与规范化
  backends/                 CUDA / Metal / HIP / CPU 的实际实现
  components/               HTTP、JSON、tokenizer、文件 I/O 等
  tests/
    server/

MFQStudio/                  前端与 Tauri 桌面壳
packaging/                  安装包、动态库与资源布局
tests/                      跨进程、接口契约与端到端检查
docs/                       用户文档、架构与迁移计划
```

服务模块按职责逐步建立；不为目录预先生成空接口或类。Engine、formats、model_source
和各设备后端的内部布局沿用 CUDA 重构方案，服务重写不另建一份模型树或算子实现。

## 4. 服务层职责

| 模块 | 拥有的行为 | 接入推理的方式 |
| --- | --- | --- |
| `api` | 输入校验、公共协议、错误映射、事件编码、静态资源 | 调用业务模块及 Scheduler 公共接口 |
| `auth` | 密钥签发、撤销、轮换、权限与策略 | 在业务操作前校验权限 |
| `models` | 模型发现、注册、配置、实例映射、加载卸载与跨实例预算协调 | 通过公开工厂创建 Engine，管理 Engine 与 Scheduler 的配对生命周期 |
| `sessions` | 持久化对话、分支、回复、幂等与版本冲突 | 组装输入，提交请求，保存结果；不新增推理准入队列 |
| `jobs` | 可持久化任务及其进度、取消、失败和重启恢复 | 调用对应原生能力；不负责逐 token 调度 |
| `hub` | 仓库元数据、文件选择、下载校验、续传和下载目录 | 下载完成后注册产物，不参与模型执行 |
| `media` | 上传、媒体引用、文档文本提取与资源访问 | 提供受校验的媒体数据或句柄 |
| `evaluations` | 数据集与结果管理、评测配置和结果对比 | 复用原生评分、推理及评测能力 |
| `cluster` | 远端节点、连接、能力发现与请求适配 | 访问真正的远端服务，保留必要的网络边界 |
| `storage` | 数据库连接、schema、事务与持久化操作 | 不持有 Engine，也不在推理期间保持写事务 |

`api` 中保留薄的路由处理。业务规则放到对应模块，不重新建立包揽所有功能的
`ServerService`。JSON 的解析和编码集中在协议及存储边界，内部优先传递已有的
C++ 请求、结果和值类型。

`server/models` 管理模型实例与服务级资源协调；Engine 管理本实例的物理权重、
KV、执行与缓存预算。复用已有资源账本和 lease，明确实际分配的唯一记账者，
避免不同实例各自按照同一份空闲内存承诺容量。

聊天记录归 `server/sessions`；模型状态、KV 和前缀快照归 Engine。两者的 session
标识可以关联，但不能把数据库里的对话状态当成设备状态。

## 5. Scheduler 与 Engine 的请求记录

| 位置 | 记录内容 | 存活范围 |
| --- | --- | --- |
| Scheduler 请求记录 | 输入、排队状态、deadline、取消意图、输出队列与待交付事件 | 接收请求到终止事件进入结果通道 |
| Engine 执行记录 | token 进度、准备任务、模型状态、在途计算、资源预留与清理 | 执行准入到物理清理或状态转存完成 |

两份记录表达不同事实，按稳定的 `RequestId` 关联：

1. 排队阶段只有 Scheduler 记录；输入和请求数量受预算约束。
2. Engine 接受准入后接管输入，Scheduler 保留调度与交付所需信息。
3. Engine 清理完请求资源，或将状态转交给自己持有的 session/cache 后，返回终止事件。
4. Scheduler 交付终止事件并退出管理；客户端持有的结果通道可以继续读取已入队事件。

这里必须区分执行完成与输出交付完成，不能用一个布尔值替代两者，也不能仅因为
两边都有请求映射就认定它们重复。

未准入请求的取消或拒绝可以由 Scheduler 直接结束。已准入请求以 Engine 的终止
确认为依据；异常路径同样要完成清理或明确强制释放，不能只调用一次 `step()`、
丢弃结果，就在外层宣布结束。每个请求对外恰好产生一个终止结果。

Duplex 的生成应逐步纳入同一请求、取消、deadline、背压与终止链路。
session 初始化和停止可以保留控制命令，不继续扩展 Scheduler 与 Engine 各自维护
active/pending/stop 的第二套生成生命周期。具体修改与 CUDA 分支协调落地。

## 6. 协议、持久化与依赖

外部能力继续包括 Studio 的 `/api/v1`、OpenAI/Anthropic 兼容接口、健康与实例身份、
SSE 和 WebSocket。兼容协议集中在 `server/api`，不与原生推理入口各维护一套。
鉴权、权限、实例身份核对、参数约束、错误结构、取消和断连处理随调用方一起迁移。

现有 OpenAPI 文件作为盘点基线。完成迁移后契约归 `server/api/openapi.json`，
维护和交付不依赖 FastAPI/Pydantic 在产品运行时生成。HTTP 兼容接口、WebSocket、
CLI 和桌面握手另行列入功能清单，不能只以 OpenAPI 路径数量判断迁移完成。

持久化继续使用 SQLite，优先沿用现有数据结构。数据库、媒体目录和资源标识的
变化必须有显式迁移；不静默覆盖、重建或丢弃现有用户数据。旧 C++ 内部接口无需
兼容，与保留用户数据和公共产品行为是两件事。

HTTP、JSON、tokenizer 和文件 I/O 优先复用现有组件；数据库使用 SQLite 原生接口。
下载组件的 HTTPS、代理、重定向、范围请求和证书能力在接入时验证，再决定依赖，
不提前引入一套新的服务框架。

依赖方向保持明确：

- 应用入口通过公开工厂装配服务和后端。
- 服务使用 Scheduler、模型源和格式模块的公开接口，不 include 后端算子或 Engine 私有实现。
- Scheduler 通过 Engine 公共接口驱动执行。
- Engine 和后端不依赖 HTTP、SQLite、桌面壳或聊天业务。
- 通用媒体解码与 tokenizer 转换归可复用原生模块；模型特有预处理语义归相应模型。

移除本地推理 IPC 不意味着移除远端通信或安全隔离。远端节点仍走网络；需要隔离的
评测程序继续在沙箱中运行，不把不可信代码直接放入服务进程。

## 7. 与现有实现的迁移对应

| 现有位置 | 目标归属 |
| --- | --- |
| `mfq/server/api/`、`protocol/` 的公共协议 | `server/api/` 与 `auth/` |
| `mfq/server/state/storage.py` | `server/storage/` 与业务模块的持久化操作 |
| `mfq/server/state/catalog.py` | `server/models/`，复用原生格式与模型源读取 |
| `mfq/server/runtime/` 的进程、端口与私有客户端 | 本地部分删除，实例生命周期归 `server/models/` |
| 同目录的配置、tokenizer、预算与远端逻辑 | 按职责归模型管理、原生文本模块、资源所有者及 `cluster/` |
| `mfq/server/services/` | 按业务迁到对应服务模块 |
| Python 媒体和文档处理 | 原生媒体模块；模型语义继续归模型与 Engine |
| 原生 `transport/` 中的本地协议 | 删除；有用的公共协议编码和校验归 `server/api/` |
| `mfq/commands/serve.py` 与桌面 Python 启动路径 | 原生应用入口及 Studio 启动逻辑 |
| 产品内的量化、校准和评测算法 | 对应原生能力，服务任务只负责调用与管理 |

旧实现只在对应功能、调用方、持久化和发布路径迁移完成后删除。研究脚本的最终位置
与依赖环境留待范围确认，不在本次架构文件中预先决定。
