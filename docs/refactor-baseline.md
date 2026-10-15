# CUDA 重构基线（阶段 0）

日期：2026-10-15。对应 [refactor.md 阶段 0](../../refactor.md#阶段-0固定基线与切换条件) 与 [refactor-plan.md §1](../../refactor-plan.md)。
本文件只记录"当前实际跑到什么结果"，不记录目标。重构过程中每完成一个阶段，就在对应行追加新链的结果，旧链结果保留到该 family 删除为止。

对照基线 commit：`4bce8b5c cuda: name the family operations for what they do, not for the seam`。
构建：`cmake --build build -j 8`（已有 build 目录，`MFQ_CUDA_ARCHITECTURES=86`、`MFQ_BUILD_METAL_RUNTIME=OFF`、`MFQ_BUILD_RUNTIME_COMMUNICATION=ON`、CUDA 13.2）。

## 环境事实（影响哪些 gate 能真跑）

| 项 | 实测 |
| --- | --- |
| GPU | RTX 4090 48GB，单卡。采集时 `strata` 与 `strata-vision` 常驻占用 44.8GB，可用约 4.3GB |
| 主机 | 32GB（`free -g` total 30），24 核 |
| 后果 | 需要并发两个完整 Engine 的 `--check-engine-isolation` 与需要 expert streaming 工作区的 `mfq-cuda-runtime-state-test` 在本机无法通过，记为缺环境而不是缺陷 |

采集时 GPU 占用是外部进程造成的，不是 MFQ 的。重跑本表前先看 `nvidia-smi --query-compute-apps=pid,used_memory --format=csv`，把占用者记进表里。

## CTest 基线

`ctest --test-dir build -j 4`：**80/82 通过**，27.83s。两个失败都是既有失败，与重构无关：

| # | 测试 | 现象 | 判定 |
| --- | --- | --- | --- |
| 29 | `mfq-float-projection-test` | 前四组 GR 融合用例 PASS，最后 `float projection: fused residual requires original up projection unsplit`（`kernels/gated_residual_fused.cu:614` 的 `splits(up)==1` 检查） | 真实缺陷，重构期间不得让它变得更糟；修它不属于本重构范围 |
| 31 | `mfq-cuda-runtime-state-test` | `CUDA VRAM leaves no workspace for expert streaming`（`storage/moe/expert_cache.cpp:67`，`device_free <= workspace_reserve_bytes()`） | 缺环境：GPU 被外部进程占用 |

三个边界检查全部通过：`mfq-cuda-include-graph-test`、`mfq-engine-include-graph-test`、`tests/test_layer_independence.py`。

## Qwen3.5-0.8B（family `qwen35`，阶段 1/2 主验收对象）

权重：`~/models/Qwen/Qwen3.5-0.8B`，raw HF safetensors，`model_type=qwen3_5`，`architectures=[Qwen3_5ForConditionalGeneration]`，1.7GB。加载日志显示 24 层 = **18 层 `linear_attention`（GDN）+ 6 层 `full_attention`**，即混合状态模型，正是阶段 2 要验证的结构。

输入 trace（固定 token id，`tokenizers` 对 `"The capital of France is, and the largest ocean is the"` 的编码，12 tokens）：

```text
760,6511,314,9338,369,11,321,279,7526,17415,369,279
```

采样档：`mfq-diagnostics` 固定 greedy（`temperature=0, top_k=1`，见 `commands/diagnostics.cpp:48`）。所有命令在 `cuda/` 目录执行，`M=$HOME/models/Qwen/Qwen3.5-0.8B`，`IDS` 为上面的列表。

| 功能 | 结果 | 命令 | 记录值 |
| --- | --- | --- | --- |
| 加载 + greedy decode | 通过 | `./build/mfq-diagnostics --model $M --ids $IDS --gen 32 --ctx-size 1024` | `prefill_sec=0.317894`，`decode_tokens=31`，`decode_tok_per_s=105.85`，RSS 峰值 1,329,160 KB |
| 前 32 个输出 token | 通过 | 同上 | `21678,20915,13,198,760,6511,314,9338,369,11,321,279,7526,17415,369,279,21678,20915,13,198,760,6511,314,9338,369,11,321,279,7526,17415,369,279`（base 模型重复输入，作为确定性 trace 使用） |
| 长 decode + 大 ctx | 通过 | `--ids $IDS --gen 128 --ctx-size 4096` | `prefill_sec=0.233092`，`decode_tokens=127`，`decode_tok_per_s=174.214`，**GPU 增量峰值 2863 MiB**（采样间隔 0.3s，基线占用 44839 MiB） |
| prefill 吞吐 | 通过 | `--ids $IDS --prefill-sweep 128,512,1024 --prefill-repeat 3 --ctx-size 1024` | m=128 `8.47ms / 15109 tok/s`；m=512 `19.94ms / 25674`；m=1024 `35.89ms / 28533` |
| 交错 batch、CUDA Graph、prefix 复用、prefill 取消、slot 复用 | 通过 | `--check-continuous-batching --ctx-size 1024` | `state_check paged=1 captures=6 prewarms=6 replays=59 prefill_cancel=1 slot_reuse=1`；`PASS requests=2 paused_ticks=4 cancellation=1 session_reuse=1` |
| session 状态声明 | 通过（弱） | `--check-text-session-state --ctx-size 1024` | `text_session_state_check dsv4=1 glm_dsa=1`。只检查声明，不验证 Qwen35 混合状态的 save/restore 数值，阶段 2 必须补 |
| MTP 正确性 | 通过 | `--check-qwen35-mtp --ctx-size 1024` | `dense_gate_cases=36 PASS`、`dense_decode_geometry_cases=15 PASS`、`PASS full_chain=1 greedy_tokens=96 stochastic_smoke_tokens=8` |
| MTP 性能（greedy，合成 id） | 通过但接受率极低 | `--bench-qwen35-mtp mtp --gen 64 --ctx-size 512` 与 `... ordinary ...` | mtp：`ttft_ms≈8.5`，`decode_tps≈166.9`，`cycles=22 accepted=0 rejected=3`，`graph_captures=0`。ordinary：`ttft_ms≈10.1–18.6`，`decode_tps≈142–167`，`graph_captures=1`。合成输入下 MTP 几乎不命中，不能当作 MTP 收益证据 |
| decode attention 数值 | 通过 | `--check-attention-decode 512 --ctx-size 1024` | `mode=full head_dim=256 length=512 parts=4`，`speedup=5.51`，`rel=4.27e-4`，`max_abs=1.90e-4`，`test_finite=1` |
| 双 Engine 隔离 | 缺环境 | `--check-engine-isolation --ctx-size 1024` | `CUDA allocation failed: requested=4718592 allocated=3362445824`。需要 GPU 空闲 ≥ 2×模型驻留后重跑 |
| CPU/SSD offload（本模型） | 未测 | `-ngl` / `--cpu-offload-layers` 组合 | 0.8B 模型驻留远小于显存，offload 意义在 27B/157GB 上，见下 |

## 其余 family 的状态

| family / 类型 | 本机权重 | 状态 |
| --- | --- | --- |
| `qwen35` → `Qwen35CausalLm` | `Qwen3.5-0.8B`（1.7GB）、`Qwen3.8-27B-FP8`（29GB） | 0.8B 见上表；27B 未跑，32GB 主机需 `-ngl`/offload 与显存空闲同时满足 |
| `qwen4_exp` → `Qwen4CausalLm` | `Qwen3.8-Flash-Next-FP8`（173GB） | 缺环境：主机内存与显存不足以常驻；`--check-flash-next`、`--check-flash-next-mtp`、`--check-qwen4-decode-hidden` 未运行 |
| `deepseek_v4` → `DeepseekV4CausalLm` | `DeepSeek-V4-Flash-Vision-Exp`（157GB，43 层 hidden 4096） | 缺环境：需 SSD 驻留路径与足够主机内存；`--check-dsv4-attention`（`docs/todo.md` 记有已知错误）、`--check-dsv4-hc` 未运行 |
| `deepseek_v41` → `DeepseekV41CausalLm` | 无 | 缺权重。仅 `mfq-deepseek-v41-config-test`、`mfq-deepseek-v41-hc-test`、`mfq-deepseek-v41-runtime-check` 通过（fixture 级） |
| `gemma4` → `Gemma4CausalLm` | 无 | 缺权重。`--check-gemma4-swa` 未运行 |
| `glm_dsa` → `GlmDsaCausalLm` | 无 | 缺权重。`--check-glm-dsa` 未运行 |
| `glm5_next` → `Glm5CausalLm` | 无 | 缺权重 |
| `minicpmo45` → `MiniCPMO45CausalLm` / `MiniCPMOTtsCausalLm` | 无 | 缺权重。阶段 6 的多模态、TTS、Duplex 全部保持未验收，不得写成通过 |

## 重跑方法

阶段 0 的采集脚本没有入库（一次性）。按上表命令逐条重跑即可；GPU 峰值用下面的采样方式：

```bash
BEFORE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
./build/mfq-diagnostics --model $M --ids $IDS --gen 128 --ctx-size 4096 & PID=$!
PEAK=$BEFORE; while kill -0 $PID 2>/dev/null; do
  U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$U" -gt "$PEAK" ] && PEAK=$U; sleep 0.3
done; wait $PID; echo "gpu_delta_mib=$((PEAK-BEFORE))"
```

主机峰值用 `/usr/bin/time -v` 的 `Maximum resident set size`。两者都要在记录里注明当时的外部占用，否则不可比较。

## 阶段 0 退出条件核对

- 基线可复现：命令与固定输入已写入本文件，CTest 80/82 与两个失败已定位到源码行。
- 缺口明确：缺权重 family 6 个、缺环境项 3 个（双 Engine、Flash-Next、DeepSeek-V4 真实权重），全部单列。
- 未删除任何运行路径：本阶段只新增本文件。

## 阶段 1（CUDA）已落地的结构

基线 `4bce8b5c` 之后，CUDA 侧的阶段 1 结构变更（`951e9bcd..d78a3ee6`）：

| 变更 | 结果 |
| --- | --- |
| 删除 `engine/internal/` 抽屉 | 按职责分为 `engine/{include,src,requests,batch,state,weights,executor,text}`，头文件名去掉目录前缀（`table.h`、`policy.h`、`driver.h`、`budget.h`…） |
| 新增 `engine/weights/` | `slots.h/.cpp`（slot 身份、lease、arena 预算）、`ranker.h`（驻留排序）、`profile.h`（驻留输入的 profile 解析）。原 `mfq-cache-policy` 目标删除，由 `mfq-engine` 编译 |
| Store 所有权 | `CudaExecutionContext` 用 `CudaStores` 持有 `moe_expert_cache`、`qsa_kv_store`；`CudaEngineState` 创建 family 的物理 batch storage，`CudaBatchExecution` 按引用借用，不再自己拥有 KV 池 |
| `backends/cuda/native/` → `device/` | 含 `tests/native/` → `tests/device/`；编译该目录的库改名 `mfq-cuda-device`。`mfq-cuda-native-kernels` 保留原名（它编译 `kernels/`） |

后端层序不变：`device -> kernels -> ops -> storage -> models -> engine`，跨目录 include 必须带目录前缀。

依赖检查（每条规则都带正/反例 fixture，规则失效会单独失败）：

```bash
python3 tests/test_engine_include_graph.py   # engine 不含 backend；模块方向受限；DAG；私有头都有消费者
python3 tests/test_cuda_include_graph.py     # 后端 DAG，跨目录 include 全部限定
python3 tests/test_layer_independence.py     # 共享层不 include engine 私有头
```

构建与测试：`cmake --build build -j16` 通过；`MFQ_BUILD_RUNTIME_COMMUNICATION=OFF` 的 `build-nocomm` 通过；CTest 仍为 80/82，两个失败与基线相同（`mfq-float-projection-test` 真实缺陷、`mfq-cuda-runtime-state-test` 显存被外部进程占用 44.8GB）。`mfq-engine` 只链接 `mfq-core`（及可选 tokenizer/prefix-cache），不链接设备库。

未做：`EngineImpl<B>` 公共装配骨架。现在没有任何消费者使用它，写出来就是死代码；它应随阶段 2 的 Qwen3.5 混合链路一起落地，由真实调用方约束形状。
