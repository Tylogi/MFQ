/** Verify generation request session isolation, event ordering, business terminal states, and server error propagation. */
import { describe, expect, it, vi } from 'vitest';
import { ApiError } from '../src/shared/api/client';
import { streamResponse } from '../src/shared/api/responses';
import type { StreamRequest } from '../src/shared/api/types';

const request: StreamRequest = {
  request_id: 'request-1',
  expected_revision: 0,
  input: [],
  sampling: {
    max_tokens: 128,
    temperature: 0.7,
    top_p: 0.9,
    top_k: 40,
    presence_penalty: 0,
    frequency_penalty: 0,
    repetition_penalty: 1,
    enable_thinking: false,
    enable_vision: false,
    enable_mtp: false,
  },
  include_reasoning_history: false,
  stream: true,
};

function frame(payload: Record<string, unknown>, sequence = 0, sessionId = 'session-1') {
  return {
    protocol_version: '1.0',
    session_id: sessionId,
    sequence,
    timestamp: '2026-09-23T00:00:00Z',
    payload,
  };
}

function respondWith(events: unknown[]) {
  vi.stubGlobal(
    'fetch',
    vi
      .fn()
      .mockResolvedValue(
        new Response(events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join(''), {
          headers: { 'content-type': 'text/event-stream' },
        }),
      ),
  );
}

const delta = { type: 'response.text.delta', response_id: 'response-1', delta: 'answer' };
const completed = { type: 'response.completed', response_id: 'response-1', finish_reason: 'stop' };

describe('describes streamResponse test behavior 1', () => {
  it('verifies streamResponse test behavior 2', async () => {
    const events = [
      frame(delta),
      frame(completed, 1),
      frame({ type: 'session.state', state: 'idle', revision: 2 }, 2),
    ];
    respondWith(events);
    const onFrame = vi.fn();
    const onAccepted = vi.fn();
    await streamResponse('session-1', request, onFrame, new AbortController().signal, onAccepted);
    expect(onAccepted).toHaveBeenCalledOnce();
    expect(onFrame.mock.calls.map(([event]) => event)).toEqual(events);
    expect(fetch).toHaveBeenCalledOnce();
    expect(fetch).toHaveBeenCalledWith(
      expect.stringContaining('/sessions/session-1/responses'),
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify(request),
      }),
    );
  });

  it('verifies streamResponse test behavior 3', async () => {
    respondWith([
      frame({ type: 'response.interrupted', response_id: 'response-1', reason: 'cancelled' }),
    ]);
    await expect(
      streamResponse('session-1', request, vi.fn(), new AbortController().signal),
    ).resolves.toBeUndefined();
  });

  it('verifies streamResponse test behavior 4', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({
      error: { code: 'REJECTED', message: 'request rejected', retryable: true, details: {} },
    }), { status: 503, headers: { 'content-type': 'application/json' } })));
    const onAccepted = vi.fn();
    await expect(
      streamResponse('session-1', request, vi.fn(), new AbortController().signal, onAccepted),
    ).rejects.toThrow('request rejected');
    expect(onAccepted).not.toHaveBeenCalled();
  });

  it('verifies streamResponse test behavior 5', async () => {
    respondWith([frame(delta)]);
    await expect(
      streamResponse('session-1', request, vi.fn(), new AbortController().signal),
    ).rejects.toThrow('before response completion');
    expect(fetch).toHaveBeenCalledOnce();
  });

  it.each([
    ['session mismatch', [frame(completed, 0, 'other-session')], 'invalid response frame'],
    ['missing sequence', [frame(delta), frame(completed, 2)], 'out-of-order'],
    ['duplicate sequence', [frame(delta), frame(completed)], 'out-of-order'],
    [
      'mixed response',
      [frame(delta), frame({ ...completed, response_id: 'other-response' }, 1)],
      'mixed response identifiers',
    ],
    ['delta after terminal state', [frame(completed), frame(delta, 1)], 'after completion'],
    ['invalid delta', [frame({ ...delta, delta: 123 })], 'invalid response payload'],
    [
      'invalid tool sequence',
      [
        frame({
          type: 'response.tool_call.delta',
          response_id: 'response-1',
          index: -1,
          arguments_delta: '{}',
        }),
      ],
      'invalid response payload',
    ],
  ])('verifies parameterized behavior %s', async (_name, events, message) => {
    respondWith(events as unknown[]);
    await expect(
      streamResponse('session-1', request, vi.fn(), new AbortController().signal),
    ).rejects.toThrow(message as string);
  });

  it('verifies streamResponse test behavior 6', async () => {
    respondWith([
      frame({
        type: 'error',
        error: { code: 'MODEL_BUSY', message: 'model busy', retryable: true, details: {} },
      }),
    ]);
    const promise = streamResponse('session-1', request, vi.fn(), new AbortController().signal);
    await expect(promise).rejects.toBeInstanceOf(ApiError);
    await expect(promise).rejects.toThrow('model busy');
    expect(fetch).toHaveBeenCalledOnce();
  });
});
