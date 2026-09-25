/** 消费并校验生成 SSE，验证事件归属、顺序与业务终态。 */
import type { RealtimeFrame, StreamRequest } from './types';
import { ApiError, apiUrl, authorizedHeaders, errorFromResponse } from './client';
import { readEventStream } from './eventStream';
import { validateResponseFrame, type ResponseFrame } from './responseProtocol';

/** 发起一次生成并校验会话、序号与业务终态；不会自动重放生成请求。 */
export async function streamResponse(
  sessionId: string,
  body: StreamRequest,
  onFrame: (frame: ResponseFrame) => void,
  signal: AbortSignal,
  onAccepted?: () => void,
): Promise<void> {
  const response = await fetch(apiUrl(`/api/v1/sessions/${sessionId}/responses`), {
    method: 'POST',
    headers: authorizedHeaders({
      Accept: 'text/event-stream',
      'Content-Type': 'application/json',
    }),
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok) throw await errorFromResponse(response);
  onAccepted?.();
  let streamError: ApiError | null = null;
  let terminal = false;
  let sequence = -1;
  let responseId: string | null = null;
  await readEventStream<unknown>(
    response,
    (value) => {
      const frame = validateResponseFrame(value, sessionId);
      if (frame.sequence !== sequence + 1)
        throw new Error('MFQ Server returned an out-of-order response event');
      sequence = frame.sequence;
      if ('response_id' in frame.payload) {
        if (responseId && responseId !== frame.payload.response_id)
          throw new Error('MFQ Server mixed response identifiers');
        responseId = frame.payload.response_id;
      }
      if (terminal && frame.payload.type.startsWith('response.'))
        throw new Error('MFQ Server returned response data after completion');
      onFrame(frame);
      if (frame.payload.type === 'error') {
        streamError = new ApiError(502, { error: frame.payload.error });
        terminal = true;
      }
      if (
        frame.payload.type === 'response.completed' ||
        frame.payload.type === 'response.interrupted'
      )
        terminal = true;
    },
    signal,
  );
  if (streamError) throw streamError;
  if (!terminal) throw new Error('MFQ Server stream ended before response completion');
}
