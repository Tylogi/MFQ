# CUDA 重构基线（阶段 0）

日期：2026-10-15。对应 [refactor.md 阶段 0](../../docs/refactor.md#阶段-0固定基线与切换条件) 与 [plan.md §1](../../docs/plan.md)；已知缺陷账本见 [bugs.md](../../docs/bugs.md)。全阶段的测试契约（跑哪些入口、需要什么环境、怎么算通过）唯一维护在 [plan.md §4 全阶段测试与验收](../../docs/plan.md#4-全阶段测试与验收)；本文件只保存测量记录与历史口径，不另立测试清单。
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
| `deepseek_v4` → `DeepseekV4CausalLm` | `DeepSeek-V4-Flash-Vision-Exp`（157GB，43 层 hidden 4096） | 缺环境：需 SSD 驻留路径与足够主机内存；`--check-dsv4-attention`（已知错误记在 [bugs.md B17](../../docs/bugs.md)）、`--check-dsv4-hc` 未运行 |
| `deepseek_v41` → `DeepseekV41CausalLm` | 无 | 缺权重。仅 `mfq-deepseek-v41-config-test`、`mfq-deepseek-v41-hc-test`、`mfq-deepseek-v41-runtime-check` 通过（fixture 级） |
| `gemma4` → `Gemma4CausalLm` | 无 | 缺权重。`--check-gemma4-swa` 未运行 |
| `glm_dsa` → `GlmDsaCausalLm` | 无 | 缺权重。`--check-glm-dsa` 未运行 |
| `glm5_next` → `Glm5CausalLm` | 无 | 缺权重 |
| `minicpmo45` → `MiniCPMO45CausalLm` / `MiniCPMOTtsCausalLm` | 无 | 缺权重。阶段 6 的多模态、TTS、Duplex 全部保持未验收，不得写成通过 |

## 重跑方法

采集脚本已入库：`sh scripts/stage0-matrix.sh <build-dir> <out-dir>`。它串行跑上面全部格子，每格写入 `$OUT/results.tsv`：格名、状态（`pass` / `fail` / `missing-env` / `untested`）、退出码、主机 RSS、完整命令和关键输出，同时保留完整日志。GPU 采样（`gpu.before`/`gpu.samples`/`gpu.peak`）是整轮峰值差值，不是逐格；只有 `long_decode_rss` 有主机 RSS，其余格留空。尾延迟、logits 基线、随机采样分布三行以 `untested` 写入，不假装采集过。权重与输入用 `MFQ_MODEL`/`MFQ_IDS` 覆盖，默认就是上表的 0.8B 与 12 token trace。

2026-10-16 起脚本的口径（fixture 测试见「G0 采集脚本可信度」）：

- 输出目录必须是新建的或空的。非空时在跑任何工具之前退出 3，两轮结果不会被混进同一份 `results.tsv`。
- 退出码：`0` = 所有格子 `pass`；`1` = 至少一格 `fail`；`2` = 没有 `fail` 但存在 `untested` 或 `missing-env`，即本轮不完整、不算验收；`3` = 开跑前被拒（输出目录非空、缺 `mfq-diagnostics`）。固定的三行 `untested` 意味着脚本现在不可能返回 `0`，这是刻意的：缺证据的运行不能报全绿。
- `$OUT/provenance.txt` 记录采集脚本自身与本次真正请求的原生可执行文件及 `CMakeCache.txt` 的 SHA256、采集脚本所在仓库的 git revision 与 dirty 文件数、`CMakeCache.txt` 里存在的配置 key（本机同时有 `CMAKE_CUDA_ARCHITECTURES=75` 与 `MFQ_CUDA_ARCHITECTURES=86`，两者都是配置项，不矛盾）、完整调用命令与 `MFQ_*` 取值。HEAD 加 dirty 计数不能识别改过的脚本，所以脚本自己的哈希单独记。缓存 key 不能证明实际编译出的 arch；二进制自己的源码 revision 记 `unknown`，`MFQ_BUILD_REVISION` 只是调用方声明，脚本会原样标注 `caller-declared, not verified`。
- `nvidia-smi` 采样进程只由本脚本启动，正常退出与收到 `INT`/`TERM` 时由 trap 终止并 reap。每次查询先单独取 stdout 与退出码再判定：退出码必须为 0，输出必须是恰好一个非空整数（空输出、多行、嵌换行、非数字都算无效），通过才写一条采样数字，否则写一条 `invalid-query` 标记行；原始查询输出不会当成可信值追加进采样。基线读数同样逐条校验。只要测量有效性丢了（基线无效、查询非 0、采样里混有 `invalid-query`、没有采样），`gpu.peak` 就写 `unavailable` 并附原因。单卡假设已写明：一次读数多行（多卡）不合并、不比较。基线之后的最大值低于基线时，保留观测到的 before/peak 数字，但 `gpu_delta_mib=unavailable`，原因是运行期间整卡占用变了且无法归因到本进程；不钳到 0，也不写负增量。
- 只有 `CUDA allocation failed` 与 `CUDA VRAM leaves no workspace` 这两个已知外部显存失败记 `missing-env`；`cannot load`、`not found` 这类应用错误按真实退出码记 `fail`，前置缺失（模型目录、tokenizer、execution-scope 可执行文件）在调用前单独判定并写精确原因。
- chunked prefill 现在是第 12 格：`MFQ_TOKENIZER` 指向 tokenizer-only GGUF，缺参数就是 `missing-env`，不再靠猜。

## 旧构建的串行对照（历史操作记录，不是当前指令）

不保留任何旧构建产物，只保留复现方法。**下面这段是当时那一轮的操作记录**：当时在 `refactor/cuda` 分支上、工作树干净，当时的 HEAD 是 `97a58dc7`：

```bash
git checkout 4bce8b5c
cmake -S csrc -B build-base -DMFQ_CUDA_ARCHITECTURES=86 -DMFQ_BUILD_METAL_RUNTIME=OFF \
      -DMFQ_BUILD_RUNTIME_COMMUNICATION=ON -DBUILD_TESTING=ON
cmake --build build-base -j 12      # 退出 0
git checkout refactor/cuda          # 脚本回到工作树，旧构建目录不动
sh scripts/stage0-matrix.sh build-base "$(mktemp -d /tmp/s0-base.XXXXXX)"   # 旧版结果
sh scripts/stage0-matrix.sh build    "$(mktemp -d /tmp/s0-head.XXXXXX)"   # 新版结果
```

保留 `-j 12` 只作为当时的调用记录，不构成对编译并发的建议（口径见 plan.md §4.4）。

脚本不往非空目录里写，所以每次重跑选一个新目录（`mktemp -d`），旧结果目录留着不覆盖。

当时的对照口径：`build-base` 对应 `4bce8b5c`，`build` 对应**当时**的 HEAD `97a58dc7`；两个结果目录里的 `results.tsv` 就是对照物。旧提交里没有采集脚本，所以不能先 checkout 再跑脚本。

今天不能照抄这段：当前工作树有未提交的采集脚本改动与未跟踪的测试文件，`git checkout` 会打断这些工作。现在要重建对照，应在**隔离的干净源码与构建目录**里按固定 ref 各配一份（另建 clone 或 `git worktree add` 到工作区外），不改当前脏工作树；`build-base` 目前不存在。当前 `build` 里的二进制在重新构建并记录身份之前，源码 revision 按 `unknown` 处理，不能当作某个 ref 的产物。下面的对照表与数值保持历史口径不变。

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
| prefill 吞吐 m=128/512/1024 | 14037 / 25653 / 28521 tok/s | 14017 / 25644 / 28525 | **待复测**，见下 |
| 主机 RSS 峰值 | 1,330,624 KB | 1,330,996 KB | 一致 |
| GPU 增量峰值（ctx 4096） | 3579 MiB | 3547 MiB | 一致 |

时间类判据不能跨运行比较。同一个基线二进制连跑两次，decode_tps 从 157.8 掉到 103.3，attention `speedup` 从 5.50 变到 9.24。prefill 也不是稳定的：同一个基线二进制重跑，吞吐呰了一半。

| 输入长度 | 第一次基线 tok/s | 第二次基线 tok/s |
| --- | ---: | ---: |
| 128 | 14037.4 | 7907.59 |
| 512 | 25652.7 | 13047.4 |
| 1024 | 28520.5 | 14330.8 |

因此“prefill 吞吐几乎不动”和“外部占用只影响 decode 排队”两个断言都撤回，prefill 性能够成待复测；GPU 空闲时重采后才能进表。已保留的只有功能正确性结论（token 序列、计数器、MTP 链路、attention 误差）。

MTP bench 的 `accepted` 自身有抖动：`prompt=3 repeat=0 warmup=0` 样本在基线两次运行里分别是 `accepted=2` 和 `accepted=18`，`cycles` 都是 22。所以 `--bench-qwen35-mtp` 只能证明“跑通且 greedy 对齐”，不能当 MTP 收益证据。

## 功能矩阵（后端 × family × 功能）

状态只用：通过 / 失败 / 缺环境 / 未测，另加 缺权重 和 不支持（该后端没有这个机制）。每格只说该 family 在该后端的证据，不用其他 family 的结果或别的环境限制代替。

### CUDA × 本机有权重的 family

| family | 加载 | greedy decode | 交错 batch | chunked prefill | prefill 吞吐 | session/prefix | MTP | CUDA Graph | offload/streaming | 并行 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `qwen35` 0.8B | 通过 `load_sec=2.26–2.63` | 通过，31/127 token 逐 token 同基线 | 通过 `PASS requests=2 paused_ticks=4 cancellation=1 session_reuse=1` | 缺环境，见下节 | 待复测（重复样本差 50%） | 通过但弱：`session_reuse=1` + `mfq-paged-prefix-cache-test`、`mfq-session-snapshot-cache-test`；`--check-text-session-state` 只检查声明 | 通过（正确性 `full_chain=1 greedy_tokens=96`）；性能格不可比 | 通过 `captures=6 prewarms=6 replays=59`，ordinary bench `graph_captures=1 graph_reuses=11` | 缺环境（`mfq-cuda-runtime-state-test` 显存不足）；策略单测通过：`mfq-cpu-expert-pool-test`、`mfq-host-expert-cache-test`、`mfq-cuda-qsa-kv-offload-test` | 未测（单卡）：`--check-tp-*`、`--check-ep-moe` 未运行 |
| `qwen35` 27B-FP8 | 未测（32GB 主机 + 显存被占） | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |
| `qwen4_exp`（Flash-Next FP8，173GB） | 缺环境 | 缺环境 | 缺环境 | 缺环境 | 缺环境 | 缺环境 | 缺环境（`--check-flash-next-mtp` 未运行） | 缺环境 | 缺环境 | 缺环境 |
| `deepseek_v4`（157GB） | 缺环境 | 缺环境 | 缺环境 | 缺环境 | 缺环境 | 缺环境 | 缺环境 | 缺环境 | 缺环境（需 SSD 驻留路径） | 缺环境 |

### CUDA × 缺权重的 family（所有功能格 = 缺权重）

| family | 本机权重 | 已有的 fixture 级证据 |
| --- | --- | --- |
| `deepseek_v41` | 无 | `mfq-deepseek-v41-config-test`、`mfq-deepseek-v41-hc-test`、`mfq-deepseek-v41-runtime-check` |
| `gemma4` | 无 | `--check-gemma4-swa` 未运行 |
| `glm_dsa` | 无 | `--check-glm-dsa` 未运行 |
| `glm5_next` | 无 | 无 |
| `minicpmo45` | 无 | 无端到端证据；`mfq-grid-vision-test` 通过但只是视觉几何 fixture，不是本 family 的媒体能力 |

媒体能力（视觉/音频/TTS/Duplex）只归属 `minicpmo45`，不记在 `qwen35` 名下。

### Metal × family

Metal 共 55 个 CTest 全部要求 macOS/MLX，本机一个都跑不了：除下表标出的机制缺失外，一律未测。机制缺失按源码事实记不支持，不用它推断公共契约失败。

| family | 加载 | greedy decode | 交错 batch | session/prefix | MTP | Graph | offload | 媒体 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `qwen35` | 未测 | 未测（`mfq-metal-qwen35-causal-lm-test`、`mfq-metal-sampling-test`） | **不支持**：`mlx_engine.h` 的 `can_batch()` 返回 `false`，`execute()` 是空实现 | 有 `mlx_paged_session_codec` 与 `mfq-metal-paged-session-codec-test`，未测 | 有 `mlx_mtp`、`mfq-metal-mtp-test`、`mfq-metal-mtp-ttt-test`，未测 | **不支持**：Metal 目录 0 处 `DecodeGraph` 引用 | 有 `mlx_qsa_kv_offload` 与 `mfq-metal-ssd-expert-arena-test`，未测 | 有 `mlx_grid_vision`、`mlx_axial_patch_vision`、`mlx_multimodal`，未测 |
| `deepseek_v4` | 未测 | 未测 | 同上不支持 | 同上 | 同上 | 同上不支持 | 同上 | 未测 |
| `deepseek_v41` | 未测 | 未测 | 同上不支持 | 同上 | 同上 | 同上不支持 | 同上 | 未测 |
| `flash_next` | 未测（`mlx_qwen4_causal_lm.cpp`） | 未测 | 同上不支持 | 同上 | 同上 | 同上不支持 | 同上 | 未测 |
| `minicpmo45` | 未测 | 未测 | 同上不支持 | 同上 | 同上 | 同上不支持 | 同上 | 未测 |

### chunked prefill 为什么仍记缺环境

`mfq-cuda-execution-scope-test` 的模型分支（`prefill_chunk_size=8`、两个 engine）只在 `argc == 3` 时执行，需要 `<model> <tokenizer-gguf>`。CTest 注册命令不带参数，所以 CTest 通过不能证明该格验证过。原先记缺环境的理由是「本机没有任何带 tokenizer 元数据的 GGUF」；这个前置在 2026-10-16 已经补齐（见下节），但真实模型执行仍未跑，所以该格继续按缺环境单列；[bugs.md B02](../../docs/bugs.md) 的记录纠错已修复，不等于该格已验收。

当前卡点是显存而不是参数：该分支要同时驻留两个 Engine，采集时 GPU 被外部进程占用 44831/49140 MiB、空闲约 3.6GB，`--check-engine-isolation` 在同一占用下已经报 `CUDA allocation failed`。GPU 空闲后跑：

```bash
# 先重建 tokenizer（仅 CPU，不读权重；产物不入库，新 checkout 上重跑即可）
.venv/bin/python -c 'from mfq.server.runtime.hf_tokenizer import ensure_hf_tokenizer_gguf as e; print(e("~/models/Qwen/Qwen3.5-0.8B", "build/stage0/tokenizers"))'

# 每次采集选一个新输出目录，不覆盖旧日志
OUT=$(mktemp -d /tmp/s0-head.XXXXXX)
MFQ_TOKENIZER=build/stage0/tokenizers/Qwen3.5-0.8B-293bd68f584541786961.tokenizer.gguf \
  sh scripts/stage0-matrix.sh build "$OUT"
# 或直接单跑该格
build/cuda/mfq-cuda-execution-scope-test ~/models/Qwen/Qwen3.5-0.8B \
  build/stage0/tokenizers/Qwen3.5-0.8B-293bd68f584541786961.tokenizer.gguf
```

采集脚本现在会把「退出 0 但打印 `model_branch=skipped`」记成 `fail` 而不是 `pass`：零退出只说明没跑模型分支，不说明 chunked prefill 验证过。

矩阵里故意缺三项，不要当成已采集：

| 项 | 状态 | 原因 |
| --- | --- | --- |
| 尾延迟 | 未采集 | `--bench-qwen35-mtp` 只打印每样本 `ttft_ms/decode_ms`；9 个混合长度样本、无并发入口，算不出有代表性的 p95/p99，也不写假分位数 |
| logits 基线 | 未采集 | 采集脚本目前没有采；`mfq-diagnostics --dump-block-trace DIR` 从 `4bce8b5c` 起就能写 `logits.f32`/`trace_meta.jsonl`，所以缺的是采集，不是参考产物（旧原因串已作废） |
| 随机采样分布 | 未采集 | `mfq-diagnostics` 固定 greedy（`temperature=0, top_k=1`）；`mixed_seed_replay=1` 只证明同种子可复现，不证明接受/拒绝分布正确。CPU 的 `sample_top_k_top_p` 是宿主参考实现（MiniCPM 消费），补不上 CUDA 采样分布基线 |

## G0 采集脚本可信度与 tokenizer 前置（2026-10-16，仅宿主侧）

本节只记录采集脚本与前置产物本身，不含任何 CUDA 数值或性能验收；上面所有历史计数不变。

- 采集脚本改动：输出目录非空拒绝（退出 3）、`provenance.txt` 产物身份（含脚本自身的 SHA256）、采样进程在退出与收到信号时被终止并 reap、每条 GPU 读数按退出码与格式逐条校验（无效则 `unavailable`，单卡假设）、`missing-env` 只认两个已知显存失败、退出码 `0/1/2/3` 区分完整/失败/不完整/拒绝、新增 `chunked_prefill` 格（`MFQ_TOKENIZER`）。CLI 与 `results.tsv` 列形状不变。
- tokenizer 前置：用生产转换器 `mfq/server/runtime/hf_tokenizer.py:ensure_hf_tokenizer_gguf(~/models/Qwen/Qwen3.5-0.8B, build/stage0/tokenizers)` 生成 tokenizer-only GGUF，不读权重张量、不改模型目录。可复制的生成命令见上节「chunked prefill 为什么仍记缺环境」；产物在 `build/` 下不入库，新 checkout 上重跑该命令即可重建。
  - 产物：`build/stage0/tokenizers/Qwen3.5-0.8B-293bd68f584541786961.tokenizer.gguf`（10,941,280 字节，SHA256 `b819a45aa4d81d5f61ab4c8854469172ceea05130f9eb4b9a53b6bdb3cb9438b`），源文件摘要与校验口径记在同目录上层 `build/stage0/tokenizer-provenance.txt`；两者都在 `.gitignore` 的 `build/` 下，不入库。
  - 已校验：`GGUF.tensor_count=0`、`general.name=Qwen3.5-0.8B`、`tokenizer.ggml.pre=qwen35`、248320 个 token 槽位里每个 `tokenizer.json` vocab id 与 26 个 added token id 都对得上、`tokenizer.chat_template` 与 `chat_template.jinja` 逐字相同、`eos=248046`、`pad=248044`。这些只证明产物来自该模型目录，不证明 CUDA 端能用。
  - 未校验：没有用 `mfq-cuda-execution-scope-test` 加载过它（本轮不跑 GPU）。
- 脚本 fixture 测试：`tests/test_stage0_matrix.py`，22 项，全部用假 `mfq-diagnostics`/`mfq-cuda-execution-scope-test`/`nvidia-smi`，不碰 GPU、不碰真实模型。覆盖成功调用与真实参数、缺 tokenizer/缺 execution-scope 二进制/缺模型目录、应用错误不被误判为 `missing-env`、显存失败仍记 `missing-env`、`model_branch=skipped` 不算 pass、非空输出目录拒绝且不覆盖上一轮、GPU 遥测的七种失效（基线后的最大值低于基线、基线读数合法但查询退出码非 0、合法与非法采样混在一起、基线无效且一次读数多行即多卡、基线合法但某次查询成功却输出为空、基线合法但每次查询输出两行、查询输出是数字但退出码非 0），后三项用故意放慢的假命令保证采样真的发生；另有 `nvidia-smi` 缺失时不忙轮询（受限 PATH 只提供脚本实际用到的外部命令，并断言没有 `dirname`/`basename`/`uniq` 这类 not found 且 `collector_script` 路径与哈希正确）、收到 `SIGTERM` 后采样进程被终止并 reap、provenance 里脚本/二进制/缓存哈希与 revision 口径。
- 实测：`.venv/bin/python -m pytest tests/test_stage0_matrix.py -q` → 22 passed；`.venv/bin/python -m pytest tests/test_hf_native_models.py -q -k "tokenizer or template or fingerprint"` → 10 passed（纯宿主 fixture，不含原生/GPU 用例）。
- 未接受：真实 CUDA 数值与性能基线、chunked prefill 的真实模型执行（B02 的记录纠错已修复，但该格本身仍未验收）、尾延迟与 logits 采集（B14 仍开放）。原生硬件不可用 = 未验收，不是测试通过。

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

engine 私有头（`requests/batch/state/weights/executor/text/src`）在 `csrc/engine/` 之外共 38 个消费者，路径相对 `csrc/`：

| 消费者 | 用到的私有头 |
| --- | --- |
| `backends/cuda/diagnostics/runtime_checks.cpp` | `table.h` |
| `backends/cuda/engine/batch_execution.h` | `driver.h` `step.h` |
| `backends/cuda/engine/cuda_engine.cpp` | `flow.h` `instance.h` `table.h` |
| `backends/cuda/engine/cuda_runtime_config.h` | `settings.h` |
| `backends/cuda/engine/generation.cpp` | `flow.h` |
| `backends/cuda/engine/generation.h` | `step.h` |
| `backends/cuda/engine/load_options.cpp` | `profile.h` |
| `backends/cuda/engine/minicpmo_components.h` | `inference.h` |
| `backends/cuda/engine/mtp.cpp` | `inference.h` `policy.h` `speculate.h` |
| `backends/cuda/models/common/mtp.h` | `predictor.h` |
| `backends/cuda/models/minicpmo45/components.h` | `policy.h` `sampling.h` |
| `backends/cuda/ops/execution/cuda_sampling.h` | `sampling.h` |
| `backends/cuda/storage/session/text_cache.cpp` | `settings.h` |
| `backends/cuda/storage/session/text_cache_impl.h` | `sessions.h` |
| `backends/cuda/storage/state/batch_paged.h` | `pages.h` |
| `backends/cuda/storage/weights/moe/cached_source_internal.h` | `slots.h` |
| `backends/cuda/storage/weights/moe/expert_cache.h` | `profile.h` |
| `backends/cuda/storage/weights/moe/host_expert_cache.h` | `slots.h` |
| `backends/cuda/storage/weights/moe/residency.cpp` | `ranker.h` |
| `backends/cuda/tests/engine/batch_execution_test.cpp` | `table.h` |
| `backends/metal/models/flash_next/mlx_qwen4_causal_lm.cpp` | `settings.h` |
| `backends/metal/runtime/mlx_engine.cpp` | `tokenizer.h` |
| `backends/metal/runtime/mlx_engine.h` | `table.h` |
| `backends/metal/runtime/mlx_generation_job.h` | `step.h` |
| `backends/metal/runtime/mlx_mtp.h` | `depth.h` |
| `backends/metal/tests/grammar_fixture.h` | `constraint.h` `tokenizer.h` |
| `tests/engine/inference_test.cpp` | `inference.h` `policy.h` |
| `tests/engine/mfq_paged_kv_allocator_test.cpp` | `pages.h` |
| `tests/engine/mfq_paged_prefix_cache_test.cpp` | `paged.h` |
| `tests/engine/mfq_sampling_test.cpp` | `sampling.h` |
| `tests/engine/mfq_session_snapshot_cache_test.cpp` | `snapshot.h` |
| `tests/engine/mfq_text_emitter_test.cpp` | `emitter.h` |
| `tests/engine/mtp_policy_test.cpp` | `predictor.h` |
| `tests/engine/request_executor_test.cpp` | `flow.h` `instance.h` `settings.h` `table.h` |
| `tests/engine/shared_execution_test.cpp` | `driver.h` `policy.h` `sampling.h` `table.h` |
| `tests/engine/speculative_sequence_test.cpp` | `speculate.h` |
| `tests/moe/moe_cache_policy_test.cpp` | `slots.h` |
| `tests/moe/moe_cache_profile_test.cpp` | `profile.h` |

两个假阳性要记住：`models/glm5_next/causal_lm.h` 和 `models/qwen4_exp/causal_lm.h` include 的是 `models/common/predictor.h`，不是 `engine/requests/predictor.h`。判断依赖要看解析后的目标文件，不能只看 include 文本首段。

Python 侧不 include 任何 engine 私有头，但 `mfq/kernels/cuda/_ext.py:52` 直接编译 `csrc/backends/cuda/native/env.cpp`。之前把 `native/` 改名成 `device/` 时这条路径失效（统一 CUDA 扩展编不过），已按计划保留 `native/` 命名并撤销改名（`97a58dc7`）。以后改目录名必须同时检查这个文件。

## Metal 支持矩阵（读代码，不看目录）

- `mfq::metal::MlxEngine` 已实现公共 `mfq::engine::Engine`（`admit/cancel/step/session/control/reload/shutdown`），并在 `apps/mfq_decode_mlx.cpp:2877` 被真实装配；它消费 `table.h`、`step.h`、`depth.h`、`tokenizer.h`、`constraint.h`、`settings.h` 六个 engine 私有头。
- Metal 执行串行：`can_batch()` 返回 `false`，`execute()` 空实现，`generate()` 一次一个请求。整个 `backends/metal` 目录 0 处 `ContinuousBatch`、`BatchPlan`、`DecodeGraph` 引用。
- family 实现 5 个：`qwen35`（full / linear attention 分开）、`deepseek_v4`、`deepseek_v41`、`flash_next`（qwen4）、`minicpmo45`；`mfq_decode_mlx.cpp` 按 backbone 分派，其余 backbone 抛 `unsupported native Metal backbone implementation`。
- `docs/runtime-support.md` 的声明：Qwen3.5–3.8、Flash-Next、DeepSeek V4/V4.1、MiniCPM-o 4.5 在 Metal 是 Native C++；GLM5–5.3 与 Gemma 4 是 Reference。
- Metal 有 55 个 CTest 用例，但 `backends/metal/CMakeLists.txt` 要求 `MFQ_MLX_ROOT` 下有 `libmlx.dylib`/`mlx.metallib` 和 Apple 框架，根 CMake 直接 `MFQ_BUILD_METAL_RUNTIME requires macOS`。本机一个都跑不了，全部记未测。
- 结论：阶段 4 用 `qwen35` 做第二后端验证（Metal 有 full/GDN 两套 attention 实现，decode app 支持该 backbone）。交错 batch 与 CUDA Graph 两格在 Metal 没有对应机制，不能拿 CUDA 结果代替，也不能因为 Metal 缺这两格就说公共契约不成立。

## 阶段 0 退出条件核对（历史口径：2026-10-15/16 那一轮）

本节记录的是当时那一轮核对的证据口径，不是当前状态，也不是全阶段测试清单；测试契约见 [plan.md §4](../../docs/plan.md#4-全阶段测试与验收)。下面的计数、数值与已知缺陷状态保持不变，只把「当时成立」与「现在需要重做」分开。

- 当时的记录口径：命令、固定输入、采集脚本均已入库；CTest 80/82 与两个失败已定位到源码行。
- 可复现性现在并不完整，与上一条分开看：`build-base` 已不在工作区，旧链对照要先按「旧构建的串行对照」一节重建；`build` 里当前二进制的源码 revision 未验证——`provenance.txt` 记的是产物哈希，二进制源码 revision 记 `unknown`，`MFQ_BUILD_REVISION` 只是调用方声明；本轮没有新的真实 GPU 运行。所以「基线可复现」当下只在方法层成立，产物层不成立。
- 旧构建可串行对照（方法层）：基线 commit 的重编方法 + 上面两张对照表已记录，不保留任何旧产物；阶段 2 每完成一段就在同一组命令上重跑。
- 缺口可跟踪：缺权重 family 6 个、缺环境 4 项（双 Engine、Flash-Next、DeepSeek-V4 真实权重、expert streaming 工作区）、Metal 全部 11 格未测，均在矩阵里单列。这些缺口作为验收门槛由 plan.md §4 在各阶段按改动消费，不是阶段 0 永久欠下的清单。
- 未删除任何运行路径：本阶段只新增脚本与本文件，并撤销一个改名回归。
- 表填完只代表记录完成：上表里只有 CUDA × qwen35 一列是真跑过的，其余列不得当成已验收。

## R1：native 命名与 Python CUDA 扩展

| 项 | 状态 |
| --- | --- |
| `native/`、`tests/native/`、`mfq-cuda-native` 与引用恢复 | 完成（`97a58dc7`），普通与通信关闭构建通过 |
| 扩展源码清单存在性检查 | 完成：`tests/test_cuda_ext_sources.py` 直接读 `_ext.py` 的 `_SOURCES`，不导入 torch |
| 统一扩展真实编译 / 加载 / 代表性算子数值 | **未验收**，见下 |

统一扩展把 26 个源文件（24 个 `.cu` + 2 个 `.cpp`）交给 torch `cpp_extension`，ninja 默认按 CPU 核数并发，单个 `cicc`/`ptxas` 峰值 1–2GB，24 并发直接打穿 32GB 主机。12:01 尝试时内核 OOM killer 杀掉了正在跑的 `strata` 服务（pid 1623662，GPU 43330 MiB）和编译自身，未产出可用扩展。`torch.utils.cpp_extension.load()` 没有 `max_jobs` 参数，并发只能用 `MAX_JOBS` 环境变量控制（见 B13）。

重跑前提：那个服务不在跑，或显式限制并发（`MAX_JOBS=2 TORCH_EXTENSIONS_DIR=/tmp/... .venv/bin/python -m pytest tests/test_activation_bridge_cuda.py`）。主机只剩 30GB、且已有服务常驻时不得跑该编译。

## 阶段 1（CUDA）已落地的结构

基线 `4bce8b5c` 之后，CUDA 侧的阶段 1 结构变更（`951e9bcd..97a58dc7`）：

| 变更 | 结果 |
| --- | --- |
| 删除 `engine/internal/` 抽屉 | 按职责分为 `engine/{include,src,requests,batch,state,weights,executor,text}`，头文件名去掉目录前缀（`table.h`、`policy.h`、`driver.h`、`budget.h`…） |
| 新增 `engine/weights/` | `slots.h/.cpp`（slot 身份、lease、arena 预算）、`ranker.h`（驻留排序）、`profile.h`（驻留输入的 profile 解析）。原 `mfq-cache-policy` 目标删除，由 `mfq-engine` 编译 |
| Store 所有权 | `CudaExecutionContext` 用 `CudaStores` 持有 `moe_expert_cache`、`qsa_kv_store`；`CudaEngineState` 创建 family 的物理 batch storage，`CudaBatchExecution` 按引用借用，不再自己拥有 KV 池 |
| `backends/cuda/native/` → `device/`（已撤销） | 改名打断 `mfq/kernels/cuda/_ext.py:52` 的 `native/env.cpp`，且计划明确保留 `native/` 命名；`97a58dc7` 撤销目录、库名与引用 |

后端层序不变：`native -> kernels -> ops -> storage -> models -> engine`，跨目录 include 必须带目录前缀。

依赖检查（每条规则都带正/反例 fixture，规则失效会单独失败）：

```bash
python3 tests/test_engine_include_graph.py   # 所有消费者都判；engine 不含 backend；模块方向受限；全图 DAG；私有头都有消费者
python3 tests/test_cuda_include_graph.py     # 后端 DAG，跨目录 include 全部限定
python3 tests/test_layer_independence.py     # 共享层不命名后端；family 名只在 engine/model/<family> 与 models/；model 层可用 state/weights，不可碰 Executor
```

三个入口跑同一套规则与 fixture：上面的脚本形式、`pytest tests/test_engine_include_graph.py tests/test_cuda_include_graph.py tests/test_layer_independence.py`（24 项，每个 fixture 一个用例）、以及 CTest 里的 `mfq-engine-include-graph-test`、`mfq-cuda-include-graph-test`、`mfq-layer-independence-test`。include 解析与 engine 模块列表共用 `tests/include_graph.py`，模块列表从目录树读取，新增模块不会被漏掉。

构建与测试：`cmake --build build -j16` 通过；`MFQ_BUILD_RUNTIME_COMMUNICATION=OFF` 的 `build-nocomm` 通过；CTest 枚举 84 项（含三项依赖检查；其中 `mfq-cuda-include-graph-test` 在 `csrc/tests/CMakeLists.txt:12` 与 `csrc/backends/cuda/CMakeLists.txt:588` 重复注册，所以 84 条不等于 84 个唯一检查，见 B15），两个失败与基线相同（`mfq-float-projection-test` 真实缺陷、`mfq-cuda-runtime-state-test` 显存被外部进程占用 44.8GB）。`mfq-engine` 只链接 `mfq-core`（及可选 tokenizer/prefix-cache），不链接设备库。

未做：`EngineImpl<B>` 公共装配骨架。现在没有任何消费者使用它，写出来就是死代码；它应随阶段 2 的 Qwen3.5 混合链路一起落地，由真实调用方约束形状。

## R2/R3：依赖检查现在真的能拦什么

`c11dd5ba`。修的是检查本身，不是代码风格；对应 bugs.md B06–B10。

| 现在能拦住的 | 实测方式 |
| --- | --- |
| core / scheduler / transport 以任何写法 include engine 私有头 | 临时放 `csrc/core/side.cpp`（裸 `"table.h"`）与 `csrc/scheduler/side.cpp`（`"../engine/requests/table.h"`），两条都报违规 |
| 后端设备代码依赖 WeightStore 驻留规则 | 临时放 `csrc/backends/cuda/ops/probe_blindspot.cpp` 含 `#include "slots.h"`，报 `may not include weights/` |
| 公共契约与私有实现成环 | 临时让 `engine/include/engine.h` 与 `engine/requests/table.h` 互 include，报出 5 条环（已恢复） |
| 共享模型层误用 Executor / 请求表 / 文本层 / 后端头 | fixture 内四个反例；同时确认 `state/`、`weights/` 正例通过 |
| 通用请求、Planner、Store 里出现 family 分支 | fixture 里 `is_qwen35()` 与 `qwen35_slots()` 两个反例；正则改为匹配词干，`Qwen35`、`qwen3_5`、`is_qwen35` 都命中 |

放宽的部分：family 名在 `engine/model/<family>/` 与 `models/` 允许，model 层允许 `state/`、`weights/`。真树上放宽后仍只有原有 3 个例外文件命中（`engine/include/engine.h`、`engine/src/processor.cpp`、`engine/src/media.cpp`），没有新增违规。

## G1 构建入口切片（验证记录，不是数值基线）

本节只记录构建入口这一片段的验收口径，与上面的数值/原生基线分开维护；它不代表 G1 或阶段 1 整体完成，也不改动上面任何历史计数。

- 已接受行为：`mfq build` 与 `mfq serve` 的自动/托管重建只请求 `mfq-runtime`（Metal 仍只 `mfq-decode-metal`）；诊断、评估、bench 用显式 `cmake --build ... --target ...`。CUDA `model.validate` 要求显式 diagnostics 路径，不回退到 runtime；缺失时返回 `tool_unavailable`，消息含预期可执行文件路径与带占位符的 `cmake --build "<build-dir>" --config <configuration> --target mfq-diagnostics`，不从可执行文件位置反推 CMake 构建目录，也不后台补编工具。CUDA 服务不注册 Metal-only 的算子测速与 WT2 质量任务，即使 sibling `mfq-perplexity`/`mfq-eval` 存在；CUDA `mfq-eval kl` 仍是显式原生 CLI，无服务适配。Metal 生产行为与错误文案不变。
- 实测：`cuda/` 下 `.venv/bin/python -m pytest tests/test_build_cli.py tests/test_serve_cli.py tests/test_server_tool_jobs.py tests/test_wikitext_quality_jobs.py -q` → 97 passed。
- CLI dry-run 实际输出：`.venv/bin/python -m mfq.cli build --backend cuda --dry-run` 的 Build 行只有 `--target mfq-runtime` 与 `-j 24`，没有 `mfq-diagnostics`/`mfq-eval`。
- 边界口径：这 4 个套件跑的是真实 Python 控制流（`create_build_plan`、serve 后端选择、job manager/store/文件边界）；只有 CMake 子进程与原生可执行文件是 mock 或脚本替身。未编译原生代码、未跑 GPU、未跑真实模型与 Metal。
- 未接受：`MFQ_BUILD_CUDA_RUNTIME` 后端开关、G2 编译单元拆分、`EngineImpl<B>` 骨架、CUDA eval 服务适配。

## 更早的 CUDA 结构整理（历史索引）

这些整理在重构开始前已完成并验证，保留结论与证据口径，不再作为待办维护。路径按当时目录，部分已在阶段 1 改名。

- 打破 `csrc/backends/cuda` 的目录级 include 循环（native↔kernels、kernels↔ops、ops↔runtime、ops↔storage）。层级固定为 native → kernels → ops → storage/runtime → models → engine，只允许向下 include；`storage` 与 `runtime` 视为同一层。
- 消除歧义 include。跨目录一律写 `"<dir>/<name>"`，CMake 不再把 `ops/`、`storage/`、`kernels/`、`models/`、`engine/` 加入 include path，裸写法无法编译。`"moe.h"` 这类同时命中 `ops/` 和 `models/common/` 的情况已消失。
- 修掉 ops/storage 循环造成的真实缺陷：`load_mfe_cpu` 和 `make_mxfp4_range_runtime` 在 `ops/moe.cpp` 与 `storage/weight_loader.cpp` 各有一份定义（链接期重复定义，且两份实现不一致，只有 ops 版支持 NINT）。现只保留 weight_loader 版。
- 完成 CUDA 侧 ops/storage 解耦：payload 解析、CPU 行切片、packed 重排、codebook 展开和 `to_device_*`/`to_gpu_*` 全部移到 `storage/*_weight_loader.cpp`，`ops/` 不再 include `storage/`，只剩 matmul/forward/dequant。当时规模：`ops` 13 文件 6811 行，`storage` 16 文件 12377 行。
- 合并两个 MTP depth controller：`DepthController`（`mtp_depth_controller.h`）和 `CycleDepthController`（`mtp_policy.h`）暴露同样 12 个方法，`update_time`/`marginal_estimate`/`time_estimate`/`score`/`best_depth`/`best_rival`/`most_stale` 近乎逐行重复（62 条长行相同）。当时合并为一个类两种调参，`cycle_timed()` 是工厂，`GenerationPolicy` 去掉模板参数，995 行 → 789 行。同一时期记录的 Metal 未跟进：现在仍可在 `backends/metal/runtime/mlx_mtp.cpp:880` 看到 `GenerationPolicy<MlxDsparkDepthController>` 写法，而 `engine/requests/depth.h:352` 的 `GenerationPolicy` 已不是模板类——引用残留是源码可见事实；“因此 Metal 无法编译”仍是当时的记录口径，本轮没有 macOS/MLX 环境，未编译验证。见 [bugs.md B18](../../docs/bugs.md)。
- 去掉 `mtp_depth_controller.h` 对进程级 `PrefillActivity` 的隐式读取：`busy` 改成显式参数，全局计数器只在 `speculative_sequence.h` 的 decode 循环读一次。
- 拆开 `request_executor.h` 的三合一：`engine_instance.h`（`EngineInstance<Backend>` + session 命令分发）与 `request_executor.h`（`ExecutionRequest` + `RequestExecutor`）。
- `mfq-models` 的 `cxx_std_17` 改为 `cxx_std_20`（头文件已使用 `co_yield` 和 `<span>`）。
- 修掉 CUDA 13 下失效的 bench 调用：`cudaStreamGetCaptureInfo_v2` 和 `cudaGraphNodeGetDependentNodes` 签名变更。
