# 服务部署与性能设置

以下以 **Windows / Linux、单张 NVIDIA GPU、MFQ 权重**为例，推荐配置针对
Qwen3.8-Flash-Next 等需要 RAM 流式专家的模型。完整参数见 [`mfq serve`](cli/serve.md)。

## 1. 构建当前版本

准备 Git、uv、CMake 3.26+、CUDA Toolkit 和 C++ 编译器。Windows 使用安装了
“使用 C++ 的桌面开发”的 Visual Studio 2022，并在 **x64 Native Tools** 终端执行；
Linux 使用 GCC / Clang。详细要求见[构建说明](cli/build.md#backend-requirements)。

```shell
git clone https://github.com/Tylogi/MFQ.git MFQ
cd MFQ
uv sync
uv run mfq build --backend cuda --build-type Release -j 8
```

原生 CUDA 推理无需安装 PyTorch。**更新源码后重新执行构建命令**；已有的托管二进制
会被 `serve` 直接使用，单独 `git pull` 不会更新它。编译是增量的。

## 2. 启动 API 服务

Linux：

```shell
uv run mfq serve --backend cuda --model "/models/model-00001-of-00005.mfq" --context-size 32768 --max-runtime-instances 1 --port 8090 --no-web-ui
```

Windows PowerShell：

```powershell
uv run mfq serve --backend cuda --model "D:/Models/model-00001-of-00005.mfq" --context-size 32768 --max-runtime-instances 1 --port 8090 --no-web-ui
```

替换模型路径；分片模型指定第一片，并保留全部配套分片和共享表文件。
`--no-web-ui` 适合 API 部署，不需要 Node.js。需要网页界面时去掉它，按提示安装
Node.js/npm；推理后端相同。

客户端填写 OpenAI Base URL：`http://127.0.0.1:8090/v1`。
使用 `GET /v1/models` 返回的 `id` 作为模型名，分片序号不属于模型名。

默认只接受本机连接。需要局域网访问时，设置 `MFQ_SERVER_API_KEY`，再加
`--host 0.0.0.0`；客户端使用服务器实际 IP 和对应 API key。

```powershell
$env:MFQ_SERVER_API_KEY = 'replace-with-your-secret'
```

Linux 用 `export MFQ_SERVER_API_KEY='replace-with-your-secret'`。

## 3. 影响速度的几个设置

| 设置 | 当前默认行为 | 建议 |
| --- | --- | --- |
| 上下文 `--context-size` | `0`：采用已保存的模型策略，或模型声明的上下文上限；并非固定 32K | 从 `32768` 开始，按实际需求增大。过大的 KV 预算会减少专家缓存、增加 RAM 压力 |
| PP 分块 `--prefill-chunk-size` | 默认 `auto`；CUDA 按待处理输入和当前工作区预算选择，最大 `8192` | 通常省略。需要限制峰值时可设 `4096` 等较小上限；预算不足时仍会缩小 |
| GPU 专家缓存 `--moe-gpu-cache-gb` | 在受支持路径上按显存和模型自动规划 | 通常省略。`0` 表示专家全部常驻，并非关闭专家缓存；不要直接填显卡总容量 |
| 专家 RAM / SSD | RAM 能容纳完整专家集时预加载；不足时自动使用有界 RAM 缓存和 SSD 专家流式 | SSD 模式优先使用本机 NVMe。它能运行更大的模型，速度受磁盘读取影响 |
| KV | CUDA 默认预分配，KV 量化和 SSD KV 默认关闭 | 保持默认作为起点；需要长上下文时再配置[KV 内存策略](runtime/cuda-kv-memory.md) |
| 并发 | CUDA 每个模型默认 1 个请求 | 有并发需求再加 `--max-requests-per-runtime 2`；更多并发会增加 KV 和工作区需求 |
| 前缀缓存 | 默认开启 | 多轮对话保留它；SSD 前缀缓存与 SSD KV 是不同功能 |

Auto 会保留专家传输缓冲区，并在支持的整层路径中临时借用可归还的专家缓存。
8192 是单次计算块的上限，**不要求整段长输入一次放进显存**。Flash-Next 可以在内部按层
处理一个逻辑段，因此日志里的逻辑段长度可能更大。普通 decode 和 MTP 使用同一分块选择。

以上 CLI 参数应用于 `--model` 的启动加载。如果先启动空服务、稍后通过 API 加载模型，
也要在加载请求中填写上下文；分块可省略，默认 Auto，例如保存为 `load.json`：

```json
{"model":"MODEL_ID","context_size":32768,"prefill_chunk_size":"auto"}
```

```shell
curl -H "Content-Type: application/json" --data-binary "@load.json" http://127.0.0.1:8090/api/v1/models/load
```

PowerShell 使用 `curl.exe`。加载返回 `202` 和操作 ID，通过
`GET /api/v1/runtime/instances` 查看模型是否进入 `ready`。

## 4. 快路径是否需要手工开启？

**通过 `mfq serve` 加载受支持的 CUDA MoE 模型时，大部分已自动配置。**
服务会选择专家缓存、RAM 锁页、两阶段 FFN、传输复用、整层 prefill 和异步上传；
CUDA Graph、主要融合算子默认开启。通常不需要复制一长串环境变量。

直接执行 `mfq-runtime` 不会经过 Python 服务的缓存和内存规划，部分原生默认值不同。
部署应使用 `mfq serve`；`--running-executable` 仅在明确需要使用匹配的预编译运行时
时指定，避免误用另一份旧二进制。

旧的实验环境变量会覆盖服务默认配置。速度异常时，检查启动环境中是否遗留
`MFQ_MOE_PREFILL_LAYER=0`、`MFQ_MOE_PIPELINE=0`、`MFQ_RUNTIME_CUDA_GRAPH=0`、
`MFQ_MOE_RAM_PIN_GIB=0` 或各种 `MFQ_DISABLE_*`，以及是否仍开启 `MFQ_TRACE_*` /
`MFQ_PROFILE_*`。只移除自己设置过的覆盖值，不要批量清空全部 `MFQ_*`。

## 5. 检查服务和速度

```shell
curl http://127.0.0.1:8090/health
curl http://127.0.0.1:8090/v1/models
```

把实际模型 ID 填进下面的 `request.json`：

```json
{
  "model": "MODEL_ID",
  "messages": [{"role": "user", "content": "介绍一下月食形成的原因。"}],
  "max_tokens": 256,
  "temperature": 0.7,
  "enable_mtp": true,
  "mtp_max_draft_tokens": 3,
  "stream": false
}
```

```shell
curl -H "Content-Type: application/json" --data-binary "@request.json" http://127.0.0.1:8090/v1/chat/completions
```

响应中的 `mfq_metrics` 提供 `prefill_tps`、`decode_tps`、`ttft_ms`、`mtp_used` 和
`mtp_acceptance_rate`。流式请求加 `"stream_options":{"include_usage":true}`，在末尾获取指标。
MTP 在有对应权重的模型上默认允许使用，会自适应决定是否采用；示例把深度上限明确设为 3。
测试普通 decode 时设 `"enable_mtp":false`。

上面的短请求用于确认服务可用。比较 PP 时使用 **4K / 8K 的真实输入**，保持模型、上下文、
分块和 MTP 设置相同；记录前缀命中情况，分别看首轮和热轮。前缀复用减少了实际 PP 工作量，
不能与完整重算混为一项。PP 看 `prefill_tps`，首 token 等待看 `ttft_ms`，生成看 `decode_tps`。
PP 包含请求内的准备、计算和必要传输；恢复专家缓存及收尾仍计入 TTFT。
decode 不包含首 token 出来前的等待。

## 6. PP 明显偏慢时

按这个顺序检查：

1. **是否使用当前源码的 Release 运行时**，启动日志会打印选中的二进制路径。
2. **实际上下文和分块**是否为预期值，尤其是空服务后续加载时是否仍用了默认参数。
3. **专家是否转入 SSD 模式**：`moe_ssd_experts_ready` 表示 SSD 专家流式；
   `moe_residency_ready` 中的 `ram_registered_bytes` 可查看 RAM 锁页量，
   `disk_reads_sealed=1` 表示完整预加载后的专家读取已封存，PLE 等独立流式表仍可读 SSD。
4. **显存 / RAM 是否紧张**：过大的上下文、固定专家缓存、额外模型和其他进程都会占用预算。
   `runtime_prefill_plan` 显示 Auto 实际选择的计算块和传输预算；
   `runtime_prefill_chunk_budget` 表示借用专家缓存后仍需进一步缩小；先查看资源占用。
5. **是否有旧的关闭快路径或 profiling 的环境变量**，以及是否存在持续磁盘读取、PCIe 带宽不足或系统换页。

这份指南参考了 [Strata 的部署说明](https://github.com/Niko1221/Strata/blob/main/docs/INSTALL.md)
和[分块、专家缓存说明](https://github.com/Niko1221/Strata/blob/main/docs/DETAILS.md)。
参数和默认值以 MFQ 自身为准；其他后端参见[支持矩阵](runtime/support.md)和[构建说明](cli/build.md)。
