# M1 直接 DMA 与冷路由列表（2026-10-15）

承接 `mfe-prequantized-gu-20261015.md` 的 68.7478 tok/s 热轮组合；保留此前所有优化。到 70 tok/s 需减少约 0.260 ms/token，即 48 层平均约 5.42 μs/层。该换算是目标，不是测量结果。

## 成本与改动

最新可复用 profile 中，每层平均约 0.49 个路由需要传输，但传输 GU 仍按十个路由及共享专家启动完整网格。Graph 里还存在 Gate/Up、Down 两次 scatter，即使原生 DMA 或缓存已将主要字段送到最终地址。

1. `MFQ_MOE_NATIVE_DIRECT_PACKET` 默认在 M1、GPU-only decode window、分阶段传输、有效 KV loan、原生 DMA、host packet 同时成立时启用。余下的小字段直接加入最终目标地址的 DMA；未锁页的字段先复制到现有的锁页暂存区。整投影缓存、页面 loan 命中仍复用原地址。原来的两次传输就绪/取消检查保留，Graph 不再启动两次 scatter。提交前断言没有 wire 字段且满足 consume-loan 条件；native batch 失败仍使用现有的直接 `cudaMemcpyAsync` 路径。
2. `MFQ_MFE_TRANSFER_ROUTE_LIST` 默认在缓存 M1、routes<=32 且已启用传输 hidden 量化融合时启用。既有等待核读取计划时，通过一次 warp ballot 写出 `[count, original_slot...]`。不新增 GPU 核、不回读 CPU、不改变 MfeFfnBatch ABI。转移 GU 采用最多四个 worker，每 worker 对超过四个的缺失路由循环处理，保留原始 slot、点积顺序、Half 激活边界和完成计数式 hidden 量化。已提前完成 GU 的 tag 6 不重复计算；取消时 count=0；重复专家保留每个输出 slot。每次 Graph 重放都会重写 count。

可分别设为 0 关闭。M2/6、多 token 分组及其它传输方式继续使用已有分派。没有展开 NVQ 权重。

代价：小字段变为更多的 DMA 条目；缺失超过四个时，同一 worker 会执行多轮；列表读取和循环也增加指令。是否抵消两个 scatter 与空 CTA 的节省，以整模测量为准。

## 已验证

- 真实 S2-L 六层，M1/2/6，共享/无共享，缓存/部分 GU/五个冷专家：864 次改变输入、重复与无效路由的 Graph 重放输出逐字节一致。M1 每次额外核对列表的 count、原 slot 和顺序，覆盖 count=0、count=5、重复/无效 slot 及刷新后的 count。实际分派为四个 worker、320 CTA；五个冷路由覆盖 worker 的第二轮。
- M1 memcheck：288 次输出及列表核对，21.42 秒，0 errors；synccheck 同样 288 次，7.91 秒，0 errors。
- 生产管线正常 168 项、宽路由/部分 GU 串行两层 84 项，包含缓存增长、重复路由和取消，输出一致，并确认真实 skip-scatter 分派。
- 关闭投影缓存时，168 项同样通过，验证剩余字段的最终地址 DMA。第一次检查在输出比较通过后，被旧测试对“必须有映射描述表字节/缓存免 DMA 阶段”的假设拒绝；测试现接受原生 DMA 传送描述表，缓存免 DMA 检查只在显式开启投影缓存时要求。生产代码未因此放宽。
- 两项新开关关闭的宽路由/部分 GU 84 项兼容检查通过。以上均为正确性检查，未重复测旧配置吞吐。

日志在 `workspace/local-server/{direct-packet-route-list-*,mfe-transfer-route-list-production-20261015*}.log`。构建清单 `cache-direct-packet-route-list-build-20261015.json` 记录全部相关源文件和实际运行文件 SHA256。

整模 API 使用保存的 `speed-kv-transfer-hidden-fused-api-20261015.json` 对照；64K 上限、FP16 KV、SSD KV 关闭、RAM 预分配、4048 输入、普通 937 输出、自然文本采样，计速排除首 token 前延迟。要求实际观测全部八类新旧分派。结果另记，未重测旧构建。

## 整模结果

`speed-kv-direct-packet-route-list-api-20261015.json` 已完成，构建及全部源码哈希在测速前核对一致。

| tok/s | 保存的 68.75 构建 | 本次组合 | 变化 |
| --- | ---: | ---: | ---: |
| 普通冷轮 | 61.8462 | 62.9270 | +1.748% |
| 普通热轮 | 68.7478 | **70.0393** | +1.879% |
| 自然文本 MTP | 65.4902 | 66.0342 | 不单独归因 |

普通两轮请求、输出全文、937 token 均与参考完全相同。热轮相当于减少约 0.268 ms/token；两项改动一起启用，只能将该差值归给整套组合，未分别测量独立收益。此次达到了单轮 70 tok/s，尚未证明长期稳定维持。

MTP 请求相同，但本轮输出 715 token，接受率 11/17（64.71%）；参考为 766 token、13/17（76.47%），输出轨迹不同，不能用这 0.831% 的吞吐差宣称同条件 MTP 提升。

八类要求的实际分派全部出现：49 个缓存驻留层，新的四 worker 传输 GU，以及直接 DMA 跳过 scatter；没有只设置开关就认定启用。三次请求后 active=0、failed=0。对照和实际分派另存 `cache-direct-packet-route-list-api-{comparison,dispatch}-20261015.json`，GPU 频率等背景另存同测量名的 `-gpu.csv`。

当前保留全部新旧优化并默认启用本次两项，服务运行本次构建。源码、二进制、SHA256、测量及 tracked diff 已保存于 `workspace/local-server/kv-cache-direct-packet-route-list-checkpoint`；此前 68.53、68.75 检查点均保留。未再次测量旧配置。
