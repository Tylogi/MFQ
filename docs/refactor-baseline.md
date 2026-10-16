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

采集脚本已入库：`scripts/stage0-matrix.sh <build-dir> <out-dir>`。它串行跑上面全部格子，每格保留完整日志、退出码、GPU 采样（`gpu.before`/`gpu.samples`/`gpu.peak`，峰值按差值算）和 `/usr/bin/time -v` 的主机 RSS。权重与输入用 `MFQ_MODEL`/`MFQ_IDS` 覆盖，默认就是上表的 0.8B 与 12 token trace。

## 旧构建的串行对照

不保留任何旧构建产物，只保留复现方法：在 `refactor/cuda` 上切回基线 commit 重编，跑同一组命令，再切回来。

```bash
git checkout 4bce8b5c
cmake -S csrc -B build -DMFQ_CUDA_ARCHITECTURES=86 -DMFQ_BUILD_METAL_RUNTIME=OFF \
      -DMFQ_BUILD_RUNTIME_COMMUNICATION=ON -DBUILD_TESTING=ON
cmake --build build -j 12      # 退出 0
sh scripts/stage0-matrix.sh build /tmp/s0-base
git checkout refactor/cuda
```

| 项 | 基线 `4bce8b5c` | HEAD `97a58dc7` |
| --- | --- | --- |
| CTest | 80/82，失败 `#29 mfq-float-projection-test`、`#31 mfq-cuda-runtime-state-test` | 同两个失败 |

同一权重、同一 12 token 输入、greedy、同一外部占用（`1623662` 常驻 43330 MiB）下的对照：

| 判据 | 基线 `4bce8b5c` | HEAD | 判定 |
| --- | --- | --- | --- |
| greedy 32 token 序列 | 上表 trace | 逐 token 相同 | 一致 |
| long decode 127 token 序列 | 重复 4 轮输入 | 逐 token 相同 | 一致 |
| continuous batching 计数器 | `paged=1 captures=6 prewarms=6 replays=59 prefill_cancel=1 slot_reuse=1`，`readbacks=46 batches=46 mixed_seed_replay=1` | 完全相同 | 一致 |
| MTP 正确性 | `PASS full_chain=1 greedy_tokens=96 stochastic_smoke_tokens=8`，`batched_linear_layers=18 ffn_calls=342` | 相同 | 一致 |
| attention decode 误差 | `rel=4.27181e-4 max_abs=1.89751e-4` | 相同 | 一致 |
| prefill 吞吐 m=128/512/1024 | 14037 / 25653 / 28521 tok/s | 14017 / 25644 / 28525 | 一致（<0.2%） |
| 主机 RSS 峰值 | 1,330,624 KB | 1,330,996 KB | 一致 |
| GPU 增量峰值（ctx 4096） | 3579 MiB | 3547 MiB | 一致 |

时间类判据不能跨运行比较。同一个基线二进制连跑两次，decode_tps 从 157.8 掉到 103.3，attention `speedup` 从 5.50 变到 9.24，prefill 吞吐却几乎不动——外部占用只影响 decode 的排队。TTFT、吞吐、尾延迟要在 GPU 空闲时重采才进表。

MTP bench 的 `accepted` 自身有抖动：`prompt=3 repeat=0 warmup=0` 样本在基线两次运行里分别是 `accepted=2` 和 `accepted=18`，`cycles` 都是 22。所以 `--bench-qwen35-mtp` 只能证明“跑通且 greedy 对齐”，不能当 MTP 收益证据。

## 功能矩阵（后端 × family × 功能）

CUDA 列来自上面的基线/HEAD 对照；Metal 列来自源码与 `docs/runtime-support.md`，本机无 macOS/MLX，全部记未测，不得凭目录写成通过。

| 功能 | CUDA × qwen35（0.8B） | CUDA × 其余 family | Metal |
| --- | --- | --- | --- |
| 加载 | 通过，`load_sec=2.26–2.63` | 27B-FP8 未跑；flash-next、dsv4 缺环境；4 个 family 缺权重 | 未测（5 个 family 有实现） |
| greedy decode | 通过，31/127 token，序列与基线逐 token 相同 | 同上 | 未测（`mfq-metal-sampling-test`、`mfq-metal-qwen35-causal-lm-test` 需 macOS） |
| 交错 batch | 通过，`PASS requests=2 paused_ticks=4 cancellation=1 session_reuse=1` | 未测 | **不支持**：`mlx_engine.h` 的 `can_batch()` 直接返回 `false`，`execute()` 是空实现 |
| chunked prefill | 通过，`mfq-cuda-execution-scope-test`（`prefill_chunk_size=8`）；`mfq-cuda-runtime-state-test` 的“一个 chunk 后让出”用例缺环境 | 未测 | 未测 |
| prefill 吞吐 | 通过，14017 / 25644 / 28525 tok/s（m=128/512/1024） | 未测 | 未测 |
| session / prefix | 通过但弱：`session_reuse=1` + `mfq-paged-prefix-cache-test`、`mfq-session-snapshot-cache-test`；`--check-text-session-state` 只检查声明，不验证混合状态 save/restore | 缺权重 | 有 `mlx_paged_session_codec` 与 `mfq-metal-paged-session-codec-test`，未测 |
| MTP | 通过（正确性），`full_chain=1 greedy_tokens=96`；性能格不可比（接受率抖动） | 未测 | 有 `mlx_mtp`、`mfq-metal-mtp-test`、`mfq-metal-mtp-ttt-test`，未测 |
| CUDA Graph | 通过，`captures=6 prewarms=6 replays=59`；ordinary bench `graph_captures=1 graph_reuses=11` | 未测 | **该机制不存在**：Metal 目录 0 处 `DecodeGraph` 引用 |
| offload / expert streaming | 缺环境（`mfq-cuda-runtime-state-test` 显存不足）；策略单测通过：`mfq-cpu-expert-pool-test`、`mfq-host-expert-cache-test`、`mfq-cuda-qsa-kv-offload-test` | 27B / 157GB 缺环境 | 有 `mlx_qsa_kv_offload` 与 `mfq-metal-ssd-expert-arena-test`，未测 |
| 媒体（视觉/音频/TTS/Duplex） | 缺权重（minicpmo45）；`mfq-grid-vision-test` 通过但只是 fixture | 缺权重 | 有 `mlx_grid_vision`、`mlx_axial_patch_vision`、`mlx_multimodal`，未测 |
| 并行（TP/EP/多卡） | 未测（单卡）：`--check-tp-*`、`--check-ep-moe` 未运行；`mfq-tensor-parallel-test`、`mfq-host-parallel-test` 通过但只是 fixture | 缺环境 | 依赖 `libjaccl`，未测 |

矩阵里故意缺三项，不要当成已采集：

| 项 | 状态 | 原因 |
| --- | --- | --- |
| 尾延迟 | 未采集 | 现有 CLI 不输出分位延迟，只有 `total_ms/decode_ms` 样本；要补得先有并发压测入口 |
| logits 基线 | 未采集 | `mfq-eval-kl` 需要参考 logits 产物，本机没有；目前只有 token 序列和 attention 的 `rel/max_abs` |
| 随机采样分布 | 未采集 | `mfq-diagnostics` 固定 greedy（`temperature=0, top_k=1`）；`mixed_seed_replay=1` 只证明同种子可复现，不证明接受/拒绝分布正确 |

## 消费者清单

工厂宏 `MFQ_CUDA_EACH_MODEL`（定义在 `csrc/backends/cuda/engine/cuda_model_list.h`，9 个类型 / 8 个 family 头）只有 5 个展开点，阶段 2 必须一起迁：

| 展开点 | 展开成什么 |
| --- | --- |
| `engine/model_registry.h:33` | `TYPE::accepts_backbone(backbone)` → 选哪个 family 加载 |
| `engine/model_bindings.cpp:140` | `load_causal_lm<TYPE>` + `TextSessionCache::restore_steps/restore_best(TYPE&, …)` |
| `engine/generation.cpp:328` | `generate(MODEL&, DecodeGraphCache&, TextSessionCache&, …)` |
| `engine/mtp.cpp:333` | `run_mtp_generation<MODEL>(…)` |
| `eval/kl.cpp:1108` | `run_kl_eval_batched<TYPE>(…)` |

加载、session 恢复、生成、MTP、KL 评估五件事共用一次宏展开。新链只换类型名而不迁这五处，旧协议会继续留在工厂里。

engine 私有头（`requests/batch/state/weights/executor/text/src`）在 `csrc/engine/` 之外共 38 个消费者：

| 区域 | 文件数 | 用到的私有头 |
| --- | --- | --- |
| `backends/cuda/engine` | 7 | `table.h` `instance.h` `flow.h` `step.h` `settings.h` `inference.h` `policy.h` `speculate.h` `profile.h` |
| `backends/cuda/storage` | 6 | `settings.h` `sessions.h` `pages.h` `slots.h` `profile.h` `ranker.h` |
| `backends/cuda/models`、`ops` | 4 | `predictor.h` `policy.h` `sampling.h` |
| `backends/cuda/diagnostics` | 1 | `table.h`（`runtime_checks.cpp`） |
| `backends/cuda/tests` | 1 | `table.h`（`batch_execution_test.cpp`） |
| `backends/metal` | 6 | `table.h` `step.h` `depth.h` `tokenizer.h` `constraint.h` `settings.h` |
| `tests/engine`、`tests/moe` | 12 | 各私有头 + `slots.h` `profile.h` |

两个假阳性要记住：`models/glm5_next/causal_lm.h` 和 `models/qwen4_exp/causal_lm.h` include 的是 `models/common/predictor.h`，不是 `engine/requests/predictor.h`。判断依赖要看解析后的目标文件，不能只看 include 文本首段。

Python 侧不 include 任何 engine 私有头，但 `mfq/kernels/cuda/_ext.py:52` 直接编译 `csrc/backends/cuda/native/env.cpp`。之前把 `native/` 改名成 `device/` 时这条路径失效（统一 CUDA 扩展编不过），已按计划保留 `native/` 命名并撤销改名（`97a58dc7`）。以后改目录名必须同时检查这个文件。

## Metal 支持矩阵（读代码，不看目录）

- `mfq::metal::MlxEngine` 已实现公共 `mfq::engine::Engine`（`admit/cancel/step/session/control/reload/shutdown`），并在 `apps/mfq_decode_mlx.cpp:2877` 被真实装配；它消费 `table.h`、`step.h`、`depth.h`、`tokenizer.h`、`constraint.h`、`settings.h` 六个 engine 私有头。
- Metal 执行串行：`can_batch()` 返回 `false`，`execute()` 空实现，`generate()` 一次一个请求。整个 `backends/metal` 目录 0 处 `ContinuousBatch`、`BatchPlan`、`DecodeGraph` 引用。
- family 实现 5 个：`qwen35`（full / linear attention 分开）、`deepseek_v4`、`deepseek_v41`、`flash_next`（qwen4）、`minicpmo45`；`mfq_decode_mlx.cpp` 按 backbone 分派，其余 backbone 抛 `unsupported native Metal backbone implementation`。
- `docs/runtime-support.md` 的声明：Qwen3.5–3.8、Flash-Next、DeepSeek V4/V4.1、MiniCPM-o 4.5 在 Metal 是 Native C++；GLM5–5.3 与 Gemma 4 是 Reference。
- Metal 有 55 个 CTest 用例，但 `backends/metal/CMakeLists.txt` 要求 `MFQ_MLX_ROOT` 下有 `libmlx.dylib`/`mlx.metallib` 和 Apple 框架，根 CMake 直接 `MFQ_BUILD_METAL_RUNTIME requires macOS`。本机一个都跑不了，全部记未测。
- 结论：阶段 4 用 `qwen35` 做第二后端验证（Metal 有 full/GDN 两套 attention 实现，decode app 支持该 backbone）。交错 batch 与 CUDA Graph 两格在 Metal 没有对应机制，不能拿 CUDA 结果代替，也不能因为 Metal 缺这两格就说公共契约不成立。

## 阶段 0 退出条件核对

- 基线可复现：命令、固定输入、采集脚本均已入库；CTest 80/82 与两个失败已定位到源码行。
- 旧构建可串行对照：基线 commit 的重编方法 + 上面两张对照表，不保留任何旧产物；阶段 2 每完成一段就在同一组命令上重跑。
- 缺口可跟踪：缺权重 family 6 个、缺环境 4 项（双 Engine、Flash-Next、DeepSeek-V4 真实权重、expert streaming 工作区）、Metal 全部 11 格未测，均在矩阵里单列。
- 未删除任何运行路径：本阶段只新增脚本与本文件，并撤销一个改名回归。
- 表填完只代表记录完成：上表里只有 CUDA × qwen35 一列是真跑过的，其余列不得当成已验收。

## 阶段 1（CUDA）已落地的结构

基线 `4bce8b5c` 之后，CUDA 侧的阶段 1 结构变更（`951e9bcd..97a58dc7`）：

| 变更 | 结果 |
| --- | --- |
| 删除 `engine/internal/` 抽屉 | 按职责分为 `engine/{include,src,requests,batch,state,weights,executor,text}`，头文件名去掉目录前缀（`table.h`、`policy.h`、`driver.h`、`budget.h`…） |
| 新增 `engine/weights/` | `slots.h/.cpp`（slot 身份、lease、arena 预算）、`ranker.h`（驻留排序）、`profile.h`（驻留输入的 profile 解析）。原 `mfq-cache-policy` 目标删除，由 `mfq-engine` 编译 |
| Store 所有权 | `CudaExecutionContext` 用 `CudaStores` 持有 `moe_expert_cache`、`qsa_kv_store`；`CudaEngineState` 创建 family 的物理 batch storage，`CudaBatchExecution` 按引用借用，不再自己拥有 KV 池 |
| `backends/cuda/native/` → `device/`（已撤销） | 改名打断 `mfq/kernels/cuda/_ext.py:52` 的 `native/env.cpp`，且计划明确保留 `native/` 命名；`97a58dc7` 撤销目录、库名与引用 |

后端层序不变：`device -> kernels -> ops -> storage -> models -> engine`，跨目录 include 必须带目录前缀。

依赖检查（每条规则都带正/反例 fixture，规则失效会单独失败）：

```bash
python3 tests/test_engine_include_graph.py   # engine 不含 backend；模块方向受限；DAG；私有头都有消费者
python3 tests/test_cuda_include_graph.py     # 后端 DAG，跨目录 include 全部限定
python3 tests/test_layer_independence.py     # 共享层不 include engine 私有头
```

构建与测试：`cmake --build build -j16` 通过；`MFQ_BUILD_RUNTIME_COMMUNICATION=OFF` 的 `build-nocomm` 通过；CTest 仍为 80/82，两个失败与基线相同（`mfq-float-projection-test` 真实缺陷、`mfq-cuda-runtime-state-test` 显存被外部进程占用 44.8GB）。`mfq-engine` 只链接 `mfq-core`（及可选 tokenizer/prefix-cache），不链接设备库。

未做：`EngineImpl<B>` 公共装配骨架。现在没有任何消费者使用它，写出来就是死代码；它应随阶段 2 的 Qwen3.5 混合链路一起落地，由真实调用方约束形状。
