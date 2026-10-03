/** 模拟 Studio 使用的 HTTP 服务，提供可控的生成、同步失败和取消场景。 */
import type { Page } from '@playwright/test';
import type { Message, OfficialModelList, ResponseResource, Session, StreamRequest } from '../src/shared/api/types';

const createdAt = '2026-09-23T00:00:00Z';
const model = 'Studio Test Model';
const session: Session = {
  id: 'session-1',
  model,
  mode: 'text',
  state: 'idle',
  revision: 0,
  title: 'Regression conversation',
  runtime_instance_id: 'instance-1',
  created_at: createdAt,
  updated_at: createdAt,
  metadata: {},
};

const modelCapabilities = {
  architecture_family: 'test',
  source: 'test',
  features: {
    text: true,
    image_input: false,
    video_input: false,
    audio_input: false,
    audio_output: false,
    full_duplex: false,
    mtp: false,
  },
};

const catalogSource = {
  provider: 'modelscope' as const, repo_id: 'example/studio-layout-test',
  url: 'https://www.modelscope.cn/models/example/studio-layout-test', available: true,
};
const catalogConfiguration = {
  status: 'recommended' as const, recommendation: 'two_stars' as const,
  required_memory_bytes: 64 * 2 ** 30, available_memory_bytes: 96 * 2 ** 30,
  reasons: ['More than half of the published precision tiers fit fully within the detected runtime memory budget.'],
};
export const officialCatalog: OfficialModelList = {
  system: { platform: 'macOS', machine: 'arm64', backend: 'metal',
    cpu_name: 'Apple M5 Max', cpu_cores: 18, gpu_names: ['Apple M5 Max'], gpu_cores: 40,
    physical_memory_bytes: 128 * 2 ** 30, runtime_memory_budget_bytes: 96 * 2 ** 30,
    memory_pools: [{ kind: 'uma', capacity_bytes: 128 * 2 ** 30, bandwidth_bytes_per_second: 614e9 }] },
  data: ['Studio Long-Context Mixture Model', 'Studio Compact Model'].map((name, index) => ({
    id: `catalog-${index}`, name, family: 'Layout fixture', architecture: 'studio_test',
    description: 'Deterministic catalog fixture for browser layout and memory-pressure checks.',
    description_zh: '用于浏览器布局与内存压力展示检查的固定目录数据。',
    parameter_label: '30B', active_parameter_label: '3B', modalities: ['text'],
    capabilities: ['reasoning'], precision_options: ['S2-L', 'S3-L', 'S4-L'],
    license: 'Apache-2.0', supports_ssd_streaming: true,
    sources: [catalogSource], selected_source: catalogSource, revision: 'test',
    downloads: 1000, likes: 25, updated_at: createdAt, configuration: catalogConfiguration,
    variants: ['S2-L', 'S3-L', 'S4-L'].map((label, tier) => ({
      id: label, label, format: 'mfq' as const, files: [`studio-${label}.mfq`],
      byte_size: (64 + tier * 24) * 2 ** 30,
      configuration: { ...catalogConfiguration, required_memory_bytes: (64 + tier * 24) * 2 ** 30 },
    })),
  })),
};

interface MockOptions {
  /** 首次生成后的历史查询返回失败，供验证手动恢复且不重复提交。 */
  failFirstSync?: boolean;
  /** 挂起生成请求直到用户调用取消接口。 */
  waitForCancel?: boolean;
  /** 挂起生成直到测试显式释放，验证跨路由生成不会被卸载取消。 */
  holdResponse?: boolean;
  /** 首次生成明确返回 HTTP 拒绝，验证草稿与附件可重试。 */
  rejectFirstSubmission?: boolean;
}

/** 为页面注册隔离的模拟 API，返回请求计数供验证生成幂等边界。 */
export async function mockStudioServer(page: Page, options: MockOptions = {}) {
  const state = { submissions: 0, cancellations: 0, unexpected: [] as string[], requests: [] as string[], releaseResponse: () => {} };
  let messages: Message[] = [];
  let responses: ResponseResource[] = [];
  let failedSync = false;
  let releasePending: (() => void) | undefined;
  await page.addInitScript(() => {
    localStorage.setItem(
      'mfq.studio.generation.v1',
      JSON.stringify({ language: 'en', theme: 'light' }),
    );
  });
  await page.route('**/api/v1/**', async (route) => {
    const path = new URL(route.request().url()).pathname;
    const method = route.request().method();
    state.requests.push(`${method} ${path}`);
    const json = async (value: unknown) => route.fulfill({ json: value });
    if (path === '/api/v1/hub/official') return json(officialCatalog);
    if (path === '/api/v1/runtime/status')
      return json({
        runtime_state: 'ready',
        model,
        model_type: 'Test',
        model_capabilities: modelCapabilities,
        max_context: 8192,
        total_requests: state.submissions,
        uptime_seconds: 60,
        active_requests: 0,
        sampling_defaults: {},
      });
    if (path === '/api/v1/runtime/capabilities')
      return json({
        model,
        model_type: 'Test',
        model_capabilities: modelCapabilities,
        vision_available: false,
        mtp_available: false,
        duplex_available: false,
      });
    if (path === '/api/v1/runtime/models') return json({ data: [{ id: model }] });
    if (path === '/api/v1/runtime/listener') return json({ host: '127.0.0.1', port: 8090, configurable: true });
    if (path === '/api/v1/runtime/memory-policy') return json({ model_limit_bytes: null, prefix_limit_bytes: null, prefix_directory: null, actual_prefix_directory: '/data/mfq/prefix-cache' });
    if (path === '/api/v1/runtime/model-aliases') return json({ aliases: {} });
    if (path === '/api/v1/runtime/instances')
      return json({
        data: [
          {
            id: 'instance-1',
            model,
            state: 'ready',
            devices: ['cpu'],
            active_sessions: 1,
            queued_requests: 0,
          },
        ],
      });
    if (path === '/api/v1/runtime/realtime/capabilities')
      return json({ available: false, defaults: {} });
    if (path === '/api/v1/components/voice-output') return json({ available: false, ready: false });
    if (path === '/api/v1/sessions') return json({ data: [session] });
    if (path === '/api/v1/sessions/session-1')
      return json({ ...session, revision: state.submissions });
    if (path === '/api/v1/sessions/session-1/messages') {
      if (options.failFirstSync && state.submissions > 0 && !failedSync) {
        failedSync = true;
        return route.fulfill({
          status: 503,
          json: {
            error: {
              code: 'SYNC_UNAVAILABLE',
              message: 'History temporarily unavailable',
              retryable: true,
              details: {},
            },
          },
        });
      }
      return json({ data: messages });
    }
    if (path === '/api/v1/sessions/session-1/responses/cancel') {
      state.cancellations += 1;
      releasePending?.();
      return json({ accepted: true });
    }
    if (path === '/api/v1/sessions/session-1/responses' && method === 'GET')
      return json({ data: responses });
    if (path === '/api/v1/sessions/session-1/responses' && method === 'POST') {
      state.submissions += 1;
      if (options.rejectFirstSubmission && state.submissions === 1)
        return route.fulfill({
          status: 503,
          json: { error: { code: 'REJECTED', message: 'Request rejected', retryable: true, details: {} } },
        });
      const body = route.request().postDataJSON() as StreamRequest;
      if (options.waitForCancel || options.holdResponse)
        await new Promise<void>((resolve) => {
          releasePending = resolve;
          state.releaseResponse = resolve;
        });
      const answer = 'A streamed answer with **formatted content**.';
      const responseId = `response-${state.submissions}`;
      const messageId = `answer-${state.submissions}`;
      const cancelled = state.cancellations > 0;
      messages = cancelled
        ? []
        : [
            {
              id: `question-${state.submissions}`,
              role: 'user',
              parts: body.input,
              parent_id: null,
              created_at: createdAt,
            },
            {
              id: messageId,
              role: 'assistant',
              parts: [{ type: 'text', text: answer }],
              parent_id: null,
              created_at: createdAt,
            },
          ];
      responses = [
        {
          id: responseId,
          request_id: body.request_id,
          session_id: session.id,
          status: cancelled ? 'cancelled' : 'completed',
          output_message_id: cancelled ? null : messageId,
          output: cancelled ? [] : [{ type: 'text', text: answer }],
          created_at: createdAt,
        },
      ];
      const payloads = cancelled
        ? [{ type: 'response.interrupted', response_id: responseId, reason: 'cancelled' }]
        : [
            {
              type: 'response.text.delta',
              response_id: responseId,
              delta: 'A streamed answer with ',
            },
            {
              type: 'response.text.delta',
              response_id: responseId,
              delta: '**formatted content**.',
            },
            { type: 'response.completed', response_id: responseId, finish_reason: 'stop' },
          ];
      await route
        .fulfill({
          contentType: 'text/event-stream',
          body: payloads
            .map(
              (payload, sequence) =>
                `data: ${JSON.stringify({
                  protocol_version: '1.0',
                  session_id: session.id,
                  sequence,
                  timestamp: createdAt,
                  payload,
                })}\n\n`,
            )
            .join(''),
        })
        .catch(() => undefined);
      return;
    }
    if (path === '/api/v1/media' && method === 'POST')
      return json({
        media: { id: 'media-1', sha256: 'test', mime_type: 'text/plain', byte_size: 4 },
        created_at: createdAt,
      });
    if (path === '/api/v1/documents' && method === 'POST')
      return json({
        media: { id: 'media-1', sha256: 'test', mime_type: 'text/plain', byte_size: 4 },
        name: 'notes.txt', text: 'note', extractor: 'text', created_at: createdAt,
      });
    if (path === '/api/v1/models/directories')
      return json({
        current_id: 'models-dir',
        current_name: 'Models',
        current_path: '/models',
        parent_id: null,
        model_file_count: 0,
        data: [{ id: 'empty-dir', name: 'Empty directory', model_file_count: 0 }],
      });
    if (
      [
        '/api/v1/models',
        '/api/v1/jobs',
        '/api/v1/runtime/metrics',
        '/api/v1/runtime/logs',
        '/api/v1/jobs/kinds',
        '/api/v1/mcp/servers',
        '/api/v1/mcp/tools',
        '/api/v1/presets',
        '/api/v1/runtime/profiles',
        '/api/v1/artifacts/lineage',
        '/api/v1/datasets',
        '/api/v1/evaluations',
        '/api/v1/cluster/nodes',
      ].includes(path)
    )
      return json({ data: [] });
    state.unexpected.push(`${method} ${path}`);
    return route.fulfill({
      status: 404,
      json: { error: { code: 'NOT_MOCKED', message: path, retryable: false, details: {} } },
    });
  });
  return state;
}
