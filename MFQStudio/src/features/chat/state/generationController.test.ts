/** 覆盖流式批处理、同步锁、取消和过期请求隔离等用户可见行为。 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { ResponseResource, Session, StreamRequest } from '../../../shared/api/types';
import { GenerationController, type ConversationSnapshot } from './generationController';
import type { ResponseFrame } from '../../../shared/api/responseProtocol';

const request = { request_id: 'request-a', stream: true } as StreamRequest;

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<T>((success, failure) => {
    resolve = success;
    reject = failure;
  });
  return { promise, resolve, reject };
}

function persisted(status: ResponseResource['status'] = 'completed'): ConversationSnapshot {
  return {
    session: { id: 'session-a', state: 'idle', revision: 2 } as Session,
    messages:
      status === 'completed'
        ? [
            {
              id: 'answer',
              role: 'assistant',
              parts: [{ type: 'text', text: 'Hello' }],
              parent_id: null,
              created_at: '',
            },
          ]
        : [],
    responses: [
      {
        id: 'response-a',
        request_id: 'request-a',
        status,
        output_message_id: status === 'completed' ? 'answer' : null,
      },
    ] as ResponseResource[],
  };
}

function delta(text: string): ResponseFrame {
  return {
    protocol_version: '1.0',
    session_id: 'session-a',
    sequence: 1,
    timestamp: '',
    payload: { type: 'response.text.delta', response_id: 'response-a', delta: text },
  };
}

function setup() {
  const transport = deferred<void>();
  const history = deferred<ConversationSnapshot>();
  let receive!: (frame: ResponseFrame) => void;
  const onSynchronized = vi.fn();
  const onSessionState = vi.fn();
  const stream = vi.fn((_id, _body, callback, signal: AbortSignal, _onAccepted?: () => void) => {
    receive = callback;
    signal.addEventListener(
      'abort',
      () => transport.reject(new DOMException('Aborted', 'AbortError')),
      { once: true },
    );
    return transport.promise;
  });
  const synchronize = vi.fn(() => history.promise);
  const cancel = vi.fn(async () => undefined);
  const controller = new GenerationController(
    { onSynchronized, onSessionState },
    { stream, synchronize, cancel },
  );
  return {
    controller,
    transport,
    history,
    receive: (frame: ResponseFrame) => receive(frame),
    onSynchronized,
    stream,
    synchronize,
    cancel,
  };
}

afterEach(() => vi.useRealTimers());

describe('GenerationController', () => {
  it('批量提交增量且保持已发布快照不可变，终态前刷新最后的文本', async () => {
    vi.useFakeTimers();
    const ctx = setup();
    const running = ctx.controller.start('session-a', request);
    const initial = ctx.controller.getSnapshot();
    const listener = vi.fn();
    ctx.controller.subscribe(listener);
    for (let i = 0; i < 100; i++) ctx.receive(delta('a'));
    expect(initial.live?.text).toBe('');
    expect(ctx.controller.getSnapshot().live?.text).toBe('');
    await vi.advanceTimersByTimeAsync(32);
    expect(ctx.controller.getSnapshot().live?.text).toBe('a'.repeat(100));
    expect(listener).toHaveBeenCalledTimes(2);
    ctx.receive(delta('last'));
    ctx.transport.resolve();
    await Promise.resolve();
    expect(ctx.controller.getSnapshot().live?.text).toBe('a'.repeat(100) + 'last');
    ctx.history.resolve(persisted());
    await running;
    expect(ctx.controller.getSnapshot().live).toBeNull();
    expect(ctx.controller.getPhase()).toBe('completed');
  });

  it('历史同步完成前保留回答且禁止再次发送', async () => {
    const ctx = setup();
    const running = ctx.controller.start('session-a', request);
    ctx.receive(delta('Hello'));
    ctx.transport.resolve();
    await Promise.resolve();
    expect(ctx.controller.getPhase()).toBe('syncing');
    expect(ctx.controller.getSnapshot().live?.text).toBe('Hello');
    await expect(ctx.controller.start('session-a', request)).rejects.toThrow('synchronize');
    ctx.history.resolve(persisted());
    await running;
    expect(ctx.onSynchronized).toHaveBeenCalledOnce();
  });

  it('同步失败保留回答，恢复操作只读历史且不重复生成', async () => {
    const ctx = setup();
    const running = ctx.controller.start('session-a', request);
    ctx.receive(delta('Hello'));
    ctx.transport.resolve();
    ctx.history.reject(new Error('offline'));
    await running;
    expect(ctx.controller.getSnapshot()).toMatchObject({
      phase: 'failed',
      recoveryNeeded: true,
      live: { text: 'Hello' },
    });
    await expect(ctx.controller.start('session-a', request)).rejects.toThrow('synchronize');
    ctx.synchronize.mockResolvedValueOnce(persisted());
    await ctx.controller.retrySynchronization();
    expect(ctx.stream).toHaveBeenCalledOnce();
    expect(ctx.controller.getSnapshot()).toMatchObject({
      phase: 'completed',
      recoveryNeeded: false,
      live: null,
    });
  });

  it('生成 POST 被明确拒绝且历史无对应响应时不确认输入', async () => {
    const ctx = setup();
    const onAccepted = vi.fn();
    const running = ctx.controller.start('session-a', request, onAccepted);
    ctx.transport.reject(new Error('HTTP 503'));
    ctx.history.resolve({ ...persisted('failed'), messages: [], responses: [] });
    await running;
    expect(onAccepted).not.toHaveBeenCalled();
    expect(ctx.controller.getSnapshot()).toMatchObject({ phase: 'failed', recoveryNeeded: false });
  });

  it('成功响应头和历史确认只会通知一次接受', async () => {
    const ctx = setup();
    const onAccepted = vi.fn();
    ctx.stream.mockImplementationOnce((_id, _body, _onFrame, _signal, accepted) => {
      accepted?.();
      return ctx.transport.promise;
    });
    const running = ctx.controller.start('session-a', request, onAccepted);
    expect(onAccepted).toHaveBeenCalledOnce();
    ctx.transport.resolve();
    ctx.history.resolve(persisted());
    await running;
    expect(onAccepted).toHaveBeenCalledOnce();
  });

  it('响应已存在但初次同步失败时，重试同步确认后清理输入且不重发 POST', async () => {
    const ctx = setup();
    const onAccepted = vi.fn();
    const running = ctx.controller.start('session-a', request, onAccepted);
    ctx.transport.reject(new Error('network interrupted'));
    ctx.history.reject(new Error('sync unavailable'));
    await running;
    expect(onAccepted).not.toHaveBeenCalled();
    ctx.synchronize.mockResolvedValueOnce(persisted('failed'));
    await ctx.controller.retrySynchronization();
    expect(onAccepted).toHaveBeenCalledOnce();
    await ctx.controller.retrySynchronization();
    expect(onAccepted).toHaveBeenCalledOnce();
    expect(ctx.stream).toHaveBeenCalledOnce();
  });

  it('重置后旧同步结果不得回写，新请求控制器保持有效', async () => {
    const ctx = setup();
    const old = ctx.controller.start('session-a', request);
    ctx.transport.resolve();
    await Promise.resolve();
    ctx.controller.reset();
    const nextTransport = deferred<void>();
    ctx.stream.mockImplementationOnce(() => nextTransport.promise);
    const next = ctx.controller.start('session-b', { ...request, request_id: 'request-b' });
    ctx.history.resolve(persisted());
    await old;
    expect(ctx.onSynchronized).not.toHaveBeenCalled();
    expect(ctx.controller.getSnapshot()).toMatchObject({
      phase: 'submitting',
      sessionId: 'session-b',
    });
    ctx.controller.reset();
    nextTransport.resolve();
    await next;
  });

  it('取消接口失败也会终止读取，同步后保留未持久化的部分回答', async () => {
    const ctx = setup();
    ctx.cancel.mockRejectedValueOnce(new Error('cancel failed'));
    const running = ctx.controller.start('session-a', request);
    ctx.receive(delta('partial'));
    await ctx.controller.stop();
    ctx.history.resolve(persisted('cancelled'));
    await running;
    expect(ctx.cancel).toHaveBeenCalledOnce();
    expect(ctx.controller.getSnapshot()).toMatchObject({
      phase: 'cancelled',
      recoveryNeeded: false,
      live: { text: 'partial' },
      error: 'cancel failed',
    });
  });

  it('服务端仍运行时阻止再次发送，防止超时取消造成并发请求', async () => {
    const ctx = setup();
    const running = ctx.controller.start('session-a', request);
    ctx.transport.reject(new Error('disconnected'));
    ctx.history.resolve(persisted('running'));
    await running;
    expect(ctx.controller.getSnapshot()).toMatchObject({ phase: 'failed', recoveryNeeded: true });
    await expect(ctx.controller.start('session-a', request)).rejects.toThrow('synchronize');
  });

  it('无终态断流时保留可恢复文本并标识失败', async () => {
    const ctx = setup();
    const running = ctx.controller.start('session-a', request);
    ctx.receive(delta('partial'));
    ctx.transport.reject(new Error('stream ended before response completion'));
    ctx.history.resolve(persisted('failed'));
    await running;
    expect(ctx.controller.getSnapshot()).toMatchObject({
      phase: 'failed',
      recoveryNeeded: false,
      live: { text: 'partial' },
    });
  });
});
