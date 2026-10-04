/** Verify SSE parsing across byte chunks and line boundaries, including reader cleanup on cancellation. */
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

describe('describes eventStream test behavior 1', () => {
  it('verifies eventStream test behavior 2', async () => {
    const events: unknown[] = [];
    const response = eventResponse(
      ': heartbeat\r\nid: 1\r\ndata: {"text":"CJK text🙂"}\r\n\r\ndata: {"done":true}\r\n\r\n',
      true,
    );
    await readEventStream(response, (event) => events.push(event));
    expect(events).toEqual([{ text: 'CJK text🙂' }, { done: true }]);
    expect(response.body?.locked).toBe(false);
  });

  it('verifies eventStream test behavior 3', async () => {
    const onEvent = vi.fn();
    await readEventStream(
      eventResponse('event: message\ndata: {"text":\ndata: "hello"}\n\n'),
      onEvent,
    );
    expect(onEvent).toHaveBeenCalledExactlyOnceWith({ text: 'hello' });
  });

  it.each(['data: {"text":"unfinished"}', 'data: {"text":"unfinished"}\n'])(
    'verifies eventStream test behavior 4',
    async (text) => {
      const response = eventResponse(text);
      const onEvent = vi.fn();
      await expect(readEventStream(response, onEvent)).rejects.toThrow('incomplete event');
      expect(onEvent).not.toHaveBeenCalled();
      expect(response.body?.locked).toBe(false);
    },
  );

  it('verifies eventStream test behavior 5', async () => {
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

  it('verifies eventStream test behavior 6', async () => {
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

  it('verifies eventStream test behavior 7', async () => {
    await expect(readEventStream(new Response('{}'), vi.fn())).rejects.toThrow(
      'invalid streaming response',
    );
  });
});
