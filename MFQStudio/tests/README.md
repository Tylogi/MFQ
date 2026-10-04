# Studio frontend tests

Run these commands from `MFQStudio`:

```sh
npm ci
npm test
npm run typecheck
npm run build
npx playwright install chromium --only-shell
npm run test:e2e
```

`npm run test:watch` is available for local watch mode. Tests use Vitest, jsdom, and Testing Library. Test files may live beside feature modules or in this directory. Shared setup cleans the DOM, browser storage, and global stubs automatically.

## Coverage boundaries

- `navigation.test.ts`: deep links, route generation, trailing slashes, and unknown-route fallback.
- `src/features/chat/markdown/markdownText.test.ts`: escaped line-break recovery plus protection for code, JSON, and math text.
- `eventStream.test.ts`: SSE byte chunks, newline variants, malformed JSON, truncation, and cancellation cleanup.
- `streamResponse.test.ts`: sessions, sequence numbers, response IDs, business terminal states, and error propagation.
- Component and state tests under `src`: maintained by each feature module and asserted through user-observable behavior.

From the repository root, run `python -m pytest tests/test_mfq_studio_source.py -q` to verify desktop packaging and platform contracts. These legacy tests still contain source assertions; update the real implementation paths when modules move. Prefer behavior tests for new interactions.

The CI `frontend` job runs Vitest, type checking, production build, and Playwright. The separate `source-contracts` job runs only native packaging, backend, and cross-language contracts. Pure frontend behavior is owned by Vitest; Python no longer reads chat, Markdown, or frontend pages to verify business behavior. The remaining Python reads of TypeScript are limited to Tauri bridge and Rust command consistency checks.

From the repository root, install test dependencies and run the same source-contract list as CI:

```sh
python -m pip install pytest 'pydantic>=2.8.0'
python -m pytest tests/test_mfq_studio_source.py tests/test_webui_markdown_prefill.py tests/test_cpp_runtime_chat_template_source.py tests/test_cpp_runtime_minicpmo45_source.py tests/test_model_capabilities.py -q
```

Local frontend scripts should use npm: `npm test`, `npm run typecheck`, `npm run build`, and `npm run test:e2e`. CI installs from `package-lock.json`.

### Responsibility split and refactor alignment

| Check area | Vitest coverage | Remaining Python coverage |
| --- | --- | --- |
| Conversation clearing, sending, and generation isolation | `chatContracts.test.tsx`, `useConversationSessions.test.tsx`, `conversationStore.test.ts`, `generationController.test.ts` | None |
| Editing and regeneration | `useMessageActions.test.tsx`, `chatContracts.test.tsx` | None |
| Scrolling, optimistic messages, and reasoning panels | `chatScrolling.test.tsx`, `chatContracts.test.tsx` | None |
| Markdown sanitization, math, copy, GFM, and escaped line breaks | `Markdown.test.tsx`, `markdownText.test.ts`, `markdownContracts.test.tsx` | None |
| Metric calculation and display | `runtimeMetrics.test.ts`, `runtimeMetricViews.test.tsx`, `markdownContracts.test.tsx` | `test_webui_markdown_prefill.py` only keeps CUDA prefill timing |
| Model capabilities, thinking, and MTP switches | `runtimeContracts.test.tsx` | `test_model_capabilities.py` keeps Python registration and C++ capability declarations |
| Templates and context reload | `runtimeContracts.test.tsx`, `settingsReload.test.tsx`, `runtimeApi.test.ts`, `runtimeMetrics.test.ts` | `test_cpp_runtime_chat_template_source.py` keeps native templates, capability publishing, and context validation |
| Media, models, settings, runtime overview, and error isolation | `studioMedia.test.tsx`, `studioBehavior.test.tsx`, `studioContracts.test.ts`, and existing module tests | `test_mfq_studio_source.py` keeps only Tauri Rust, packaging config, release scripts, and platform bridge checks |
| Background job streams | `jobStore.test.ts`, `useJobEventLog.test.tsx`, `studioContracts.test.ts` | None |
| Voice | `audioCodec.test.ts`, `AudioDevices.test.ts`, `voiceContracts.test.ts` | `test_cpp_runtime_minicpmo45_source.py` keeps C++ implementation and Python realtime gateway checks |

Former frontend session and scrolling source checks have moved to Vitest, so duplicate Python index files are no longer kept. `tests/test_webui_markdown_prefill.py` keeps its historical filename, but its contents only cover native timing.

This split does not claim that every legacy source assertion is a behavior test. `studioContracts.test.ts` and `voiceContracts.test.ts` explicitly keep some transitional source contracts, while `markdownContracts.test.tsx` keeps dependency and style-structure constraints. Complex duplex turn ownership checks temporarily keep their existing protection strength and must not be replaced by audio codec tests. New business checks should call functions, hooks, or rendered components first; replace transitional contracts with equivalent behavior tests over time to avoid maintaining duplicate assertions.

### Browser test boundaries

Playwright builds first, then starts `vite preview` against the production output and mocks the API. It does not reuse a potentially stale development server. Desktop and mobile runs represent browser viewports, not native Tauri apps. They verify frontend interactions in the built artifact, but they do not prove that deep-link refresh works in the real MFQ static service or that real models, microphones, and speakers work in a full-duplex path. A separate service integration test owns the real SPA fallback.

Production builds do not include test files. `npm run typecheck` also checks tests and test config so assertion type errors are not missed. CI runs behavior tests, type checks, build, desktop and mobile browser regressions, and Python contract tests.

Use npm to update `package-lock.json` when development dependencies change. Do not edit lockfiles by hand.
