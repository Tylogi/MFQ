/** 验证 SSE 在字节分片、换行边界和取消情况下正确解析并释放读取器。 */
import { describe, expect, it, vi } from 'vitest';
import { readEventStream } from '../src/shared/api/eventStream';

function eventResponse(text: string, splitBytes = false) {
  const bytes = new TextEncoder().encode(text);
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      if (splitBytes) {
        for (const byte of bytes) controller.enqueue(Uint8Array.of(byte));
      } else {
        controller.enqueue(bytes);
      }
      controller.close();
    },
  });
  return new Response(stream, { headers: { 'content-type': 'text/event-stream; charset=utf-8' } });
}

describe('SSE 传输', () => {
  it('UTF-8 多字节文本与 CRLF 跨块时只派发完整事件', async () => {
    const events: unknown[] = [];
    const response = eventResponse(
      ': heartbeat\r\nid: 1\r\ndata: {"text":"中文🙂"}\r\n\r\ndata: {"done":true}\r\n\r\n',
      true,
    );
    await readEventStream(response, (event) => events.push(event));
    expect(events).toEqual([{ text: '中文🙂' }, { done: true }]);
    expect(response.body?.locked).toBe(false);
  });

  it('按 SSE 规则合并多行 data 并忽略未知字段', async () => {
    const onEvent = vi.fn();
    await readEventStream(
      eventResponse('event: message\ndata: {"text":\ndata: "hello"}\n\n'),
      onEvent,
    );
    expect(onEvent).toHaveBeenCalledExactlyOnceWith({ text: 'hello' });
  });

  it.each(['data: {"text":"unfinished"}', 'data: {"text":"unfinished"}\n'])(
    '拒绝没有事件结束符的 EOF',
    async (text) => {
      const response = eventResponse(text);
      const onEvent = vi.fn();
      await expect(readEventStream(response, onEvent)).rejects.toThrow('incomplete event');
      expect(onEvent).not.toHaveBeenCalled();
      expect(response.body?.locked).toBe(false);
    },
  );

  it('JSON 损坏时取消未结束的流并释放锁', async () => {
    const cancel = vi.fn();
    const response = new Response(
      new ReadableStream({
        start(controller) {
          controller.enqueue(new TextEncoder().encode('data: invalid\n\n'));
        },
        cancel,
      }),
      { headers: { 'content-type': 'text/event-stream' } },
    );
    await expect(readEventStream(response, vi.fn())).rejects.toThrow();
    expect(cancel).toHaveBeenCalledOnce();
    expect(response.body?.locked).toBe(false);
  });

  it('取消挂起读取时立即拒绝并释放锁', async () => {
    const cancel = vi.fn();
    const abort = new AbortController();
    const response = new Response(new ReadableStream({ cancel }), {
      headers: { 'content-type': 'text/event-stream' },
    });
    const pending = readEventStream(response, vi.fn(), abort.signal);
    abort.abort(new Error('user stopped'));
    await expect(pending).rejects.toThrow('user stopped');
    expect(cancel).toHaveBeenCalledOnce();
    expect(response.body?.locked).toBe(false);
  });

  it('拒绝非 SSE 响应', async () => {
    await expect(readEventStream(new Response('{}'), vi.fn())).rejects.toThrow(
      'invalid streaming response',
    );
  });
});
