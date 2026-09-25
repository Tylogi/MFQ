/** 模拟 Studio 使用的 HTTP 服务，提供可控的生成、同步失败和取消场景。 */
import type { Page } from '@playwright/test';
import type { Message, ResponseResource, Session, StreamRequest } from '../src/shared/api/types';

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
