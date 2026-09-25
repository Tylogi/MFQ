# MFQ Studio 前端

Web 与 Tauri 共用 React 前端，使用 npm 管理前端依赖。

## 本地运行

```sh
npm ci
npm run dev
```

Vite 默认监听 `127.0.0.1:5173`，将 `/api` 代理到 `127.0.0.1:8090`。实际推理需要启动 MFQ 服务；浏览器测试使用模拟接口，不需要模型、GPU 或凭据。浏览器使用 History 路由，例如 `/chat`；Tauri 桌面环境使用 Hash 路由，例如 `/#/chat`，兼容桌面资源协议。

## 生产环境页面路由

MFQ 主服务和实时音频网关共用 SPA 静态服务：优先返回实际文件；找不到文件时，仅对接受 `text/html` 的 GET/HEAD 页面导航返回 `index.html`，保留原 URL。新增前端页面无需在后端逐个注册，未知页面由前端显示 404 界面。

页面路径约定不带文件扩展名，也不占用 `api`、`v1`、`realtime`、`health`、`docs`、`redoc`、`assets`、`static` 等保留前缀。接口错误、缺失静态资源及非页面请求不会回退为 HTML。若在 MFQ 前面配置反向代理，需将页面请求转发给 MFQ，或配置同等规则；构建资源继续使用根路径 `/assets/...`。

部署后应直接打开并刷新 `/chat`、`/models`、`/settings`，确认页面返回 200 且地址不变，同时确认不存在的接口和静态资源仍返回错误。服务端回归测试：

```sh
python -m pytest tests/test_server_static.py -q
```

## 模块边界

| 目录 | 职责 |
| --- | --- |
| `src/app` | 导航外壳、模型工具布局、共享运行时、格式化和通用展示 |
| `src/features/chat` | 消息、附件、输入组件、生成生命周期与滚动 |
| `src/features/voice` | 实时协议状态机、独立音频设备、PCM 编解码、重采样、片段存储 |
| `src/features/settings` | 设置页面、推理配置、预设转换 |
| `src/features/models` | 模型目录、加载策略、模型仓库页面及相关操作 |
| `src/features/runtime` | 概览、缓存、日志页面和运行配置面板 |
| `src/features/jobs` | 量化页面、页面内任务状态与各业务面板 |
| `src/features/evaluations`、`connections` | 评测比较、数据集注册、MCP 与远程节点配置 |
| `src/shared/api` | HTTP、SSE、协议校验、按领域划分的契约和资源 API |
| `src/shared/ui` | Dialog、Tooltip、Switch 的应用级 Radix 封装 |
| `src/shared/platform` | Web/Tauri 桥接 |

`App.tsx` 仅组合 Provider 与懒加载路由，不持有业务请求、表单或业务操作。`StudioShell` 只负责导航与运行状态摘要，`LabLayout` 只负责模型工具的二级导航。`studio.ts`、`realtimeAudio.ts` 和 API 的 `types.ts` 保留必要的公共入口，资源请求必须按领域从 `shared/api/resources` 导入。

`RuntimeProvider` 仅初始化平台地址、凭据、实例能力和共享后台任务，不加载评测、日志、模型目录、预设等页面数据。各页面挂载后请求自己的资源，并在离开时清理订阅或忽略过期返回。`SettingsProvider` 只保存已应用偏好和跨页面上下文容量，设置草稿与预设操作归设置页面。

`ChatProvider` 仅组合聊天领域的会话、生成、附件、消息动作和语音 hook。它位于路由外以保持后台生成，首次访问聊天才加载会话与工具清单。量化页面的 context 只在量化页面内共享表单、任务操作和日志，不能被其他业务拿来作为全局状态容器。

`styles.css` 只保留有序导入；主题、布局、通用组件和各业务样式独立维护，原有覆盖规则留在最后的 `overrides.css`，避免拆文件时改变级联顺序。

局部输入草稿按会话保存于内存 Zustand store，不写入浏览器持久化存储。聊天领域只订阅生成阶段，`StreamingMessage` 订阅增量快照，历史正文通过 memo 和稳定业务回调隔离。切换其他页面不重新提交或取消正在运行的文本生成，切换服务连接则失效旧会话和请求。

## 生成生命周期

```text
submitting -> streaming -> syncing -> completed
                    |         |
                 stopping     failed -> 重新同步
                    |
                 syncing -> cancelled
```

- 每个请求保留自己的会话与请求标识；重置后旧流、旧同步和旧取消回调失效。
- SSE 校验版本、会话、连续序号、响应标识和业务终态。正常 EOF 不等于生成完成。
- 文本增量按 32ms 窗口批量提交，结束时强制刷新；Markdown 在流式阶段按 80ms 窗口解析，完成后立即刷新并补充公式、复制按钮。所有 HTML 仍经过 DOMPurify。
- 历史同步完成前保留临时回答并锁定发送；同步失败后保留收到的文本，恢复按钮只读取历史，不重放生成 POST。
- 取消请求最多等待 5 秒，历史同步最多等待 10 秒。无法确认服务端停止时保持恢复状态，防止产生重复生成。
- 滚动跟随由内容尺寸变化驱动，用户向上阅读后暂停。输入组件处理中文输入法组合状态，移动端会话侧栏默认关闭。

## 验证

```sh
npm test
npm run typecheck
npm run build
npx playwright install chromium --only-shell
npm run test:e2e
```

从仓库根目录运行：

```sh
python -m pytest tests/test_mfq_studio_source.py -q
```

单元测试覆盖架构边界、模型目录操作、设置共享、分块 SSE、协议错误、取消、历史恢复、旧请求隔离、草稿渲染边界、Markdown 净化、音频设备释放及分块等价性。浏览器测试覆盖桌面和移动端按需请求、跨页继续生成、草稿恢复、发送、取消、恢复、输入法及弹窗键盘焦点，截图与失败追踪输出到被忽略的 `artifacts/playwright`。

这些检查不替代真实模型推理、麦克风、扬声器及 Tauri 原生窗口联调。没有将合成事件测试描述为真实模型性能提升；长历史虚拟化及稳定 Markdown 分块缓存应在真实长会话压测证明必要后加入。
