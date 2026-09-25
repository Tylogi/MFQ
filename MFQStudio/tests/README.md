# Studio 前端测试

在 `MFQStudio` 目录运行：

```sh
npm ci
npm test
npm run typecheck
npm run build
npx playwright install chromium --only-shell
npm run test:e2e
```

`npm run test:watch` 用于本地持续运行。测试使用 Vitest、jsdom 与 Testing Library；测试文件既可放在业务模块旁，也可放在本目录。统一配置自动清理 DOM、浏览器存储和全局桩对象。

## 覆盖边界

- `navigation.test.ts`：深链接、路由生成、尾部斜杠及未知地址回退。
- `src/features/chat/markdown/markdownText.test.ts`：转义换行恢复与代码、JSON、数学文本保护。
- `eventStream.test.ts`：SSE 字节分片、换行、畸形 JSON、截断与取消清理。
- `streamResponse.test.ts`：会话、序号、响应标识、业务终态和错误传播。
- `src` 内的组件及状态测试：由对应业务模块维护，按用户可观察行为断言。

从仓库根目录执行 `python -m pytest tests/test_mfq_studio_source.py -q`，校验桌面打包及平台契约。这组历史测试仍包含源码断言，模块拆分时应更新真实实现路径；新交互优先用行为测试覆盖。

CI 的 `frontend` job 运行 Vitest、类型检查、构建和 Playwright；独立的 `source-contracts` job 只运行原生打包、后端及跨语言契约。纯前端测试由 Vitest 负责，Python 不再读取聊天、Markdown 或前端页面来验证其业务行为。Python 中保留的 TypeScript 读取仅用于检查 Tauri 桥接与 Rust 命令是否一致。

从仓库根目录安装测试依赖并运行与 CI 相同的清单：

```sh
python -m pip install pytest 'pydantic>=2.8.0'
python -m pytest tests/test_mfq_studio_source.py tests/test_webui_markdown_prefill.py tests/test_cpp_runtime_chat_template_source.py tests/test_cpp_runtime_minicpmo45_source.py tests/test_model_capabilities.py -q
```

本地也可使用 `pnpm test`、`pnpm run typecheck`、`pnpm run build` 和 `pnpm run test:e2e` 执行同一组前端脚本；CI 安装继续使用仓库的 `package-lock.json`。

### 检查分工与重构适配

| 检查内容 | Vitest 覆盖位置 | Python 保留内容 |
| --- | --- | --- |
| 会话清空、发送、生成隔离 | `chatContracts.test.tsx`、`useConversationSessions.test.tsx`、`conversationStore.test.ts`、`generationController.test.ts` | 无 |
| 编辑与重新生成 | `useMessageActions.test.tsx`、`chatContracts.test.tsx` | 无 |
| 滚动、乐观消息、推理面板 | `chatScrolling.test.tsx`、`chatContracts.test.tsx` | 无 |
| Markdown 净化、公式、复制、GFM、转义换行 | `Markdown.test.tsx`、`markdownText.test.ts`、`markdownContracts.test.tsx` | 无 |
| 指标计算与展示 | `runtimeMetrics.test.ts`、`runtimeMetricViews.test.tsx`、`markdownContracts.test.tsx` | `test_webui_markdown_prefill.py` 仅保留 CUDA prefill 计时 |
| 模型能力、思考与 MTP 开关 | `runtimeContracts.test.tsx` | `test_model_capabilities.py` 保留 Python 注册和 C++ 能力声明 |
| 模板、上下文重载 | `runtimeContracts.test.tsx`、`settingsReload.test.tsx`、`runtimeApi.test.ts`、`runtimeMetrics.test.ts` | `test_cpp_runtime_chat_template_source.py` 保留原生模板、能力发布和上下文校验 |
| 媒体、模型、设置、运行概览及错误隔离 | `studioMedia.test.tsx`、`studioBehavior.test.tsx`、`studioContracts.test.ts` 与现有模块测试 | `test_mfq_studio_source.py` 仅保留 Tauri Rust、打包配置、发布脚本及平台桥接 |
| 后台任务流 | `jobStore.test.ts`、`useJobEventLog.test.tsx`、`studioContracts.test.ts` | 无 |
| 语音 | `audioCodec.test.ts`、`AudioDevices.test.ts`、`voiceContracts.test.ts` | `test_cpp_runtime_minicpmo45_source.py` 保留 C++ 实现与 Python 实时网关 |

原有前端会话与滚动源码检查已迁入 Vitest，不再保留重复的 Python 索引文件。`tests/test_webui_markdown_prefill.py` 沿用历史文件名，但内容只有原生计时。

本轮完成执行归属分层，并不把所有历史源码断言伪称为行为测试：`studioContracts.test.ts` 和 `voiceContracts.test.ts` 明确保留部分过渡源码契约，`markdownContracts.test.tsx` 保留依赖及样式结构约束。复杂双工回合归属等检查暂时保持原有保护强度，不能用音频编码测试代替。新业务检查优先调用函数、hook 或渲染组件；后续逐项用等价行为测试替代过渡契约，避免同时维护重复断言。

### 浏览器测试边界

Playwright 先构建，再启动 `vite preview` 加载生产产物，并模拟 API；不会复用可能残留的开发服务器。桌面和移动端表示浏览器视口，不代表原生 Tauri 应用。它验证构建产物的前端交互，不能证明真实 MFQ 静态服务的深链接刷新正常，也不能证明真实模型、麦克风与扬声器的全双工链路正常。真实服务的 SPA fallback 由单独的服务集成测试负责。

生产构建不纳入测试文件。`npm run typecheck` 另外检查测试及测试配置，避免遗漏断言类型错误。CI 同时执行行为测试、类型检查、构建、桌面与移动端浏览器回归及 Python 契约测试。

开发依赖变动时使用 npm 更新 `package-lock.json`，不手工修改锁文件。
