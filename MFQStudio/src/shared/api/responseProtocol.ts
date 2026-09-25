/** 校验文本生成协议边界，防止外部事件污染会话或写入无效状态。 */
import type { ApiErrorBody, RealtimeFrame, SessionState } from './types';

type ResponsePayload =
  | { type: 'session.state'; state: SessionState; revision: number }
  | { type: 'response.text.delta' | 'response.reasoning.delta'; response_id: string; delta: string }
  | {
      type: 'response.tool_call.delta';
      response_id: string;
      index: number;
      arguments_delta: string;
    }
  | { type: 'response.completed'; response_id: string; finish_reason: string }
  | { type: 'response.interrupted'; response_id: string; reason: string }
  | { type: 'error'; error: ApiErrorBody['error'] };

export type ResponseFrame = Omit<RealtimeFrame, 'payload'> & { payload: ResponsePayload };

const sessionStates = new Set([
  'idle',
  'listening',
  'processing',
  'speaking',
  'interrupted',
  'reconnecting',
  'error',
  'closed',
]);

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/** 将服务端未知 JSON 校验为可消费事件；会话不匹配或字段错误时立即失败。 */
export function validateResponseFrame(value: unknown, sessionId: string): ResponseFrame {
  if (
    !isRecord(value) ||
    value.protocol_version !== '1.0' ||
    value.session_id !== sessionId ||
    !Number.isSafeInteger(value.sequence) ||
    Number(value.sequence) < 0 ||
    typeof value.timestamp !== 'string' ||
    !isRecord(value.payload)
  ) {
    throw new Error('MFQ Server returned an invalid response frame');
  }
  const payload = value.payload;
  let valid = false;
  switch (payload.type) {
    case 'session.state':
      valid =
        sessionStates.has(String(payload.state)) &&
        Number.isSafeInteger(payload.revision) &&
        Number(payload.revision) >= 0;
      break;
    case 'response.text.delta':
    case 'response.reasoning.delta':
      valid = typeof payload.delta === 'string';
      break;
    case 'response.tool_call.delta':
      valid =
        Number.isSafeInteger(payload.index) &&
        Number(payload.index) >= 0 &&
        Number(payload.index) < 1024 &&
        typeof payload.arguments_delta === 'string';
      break;
    case 'response.completed':
      valid = typeof payload.finish_reason === 'string';
      break;
    case 'response.interrupted':
      valid = typeof payload.reason === 'string';
      break;
    case 'error':
      valid =
        isRecord(payload.error) &&
        typeof payload.error.message === 'string' &&
        typeof payload.error.code === 'string' &&
        typeof payload.error.retryable === 'boolean' &&
        isRecord(payload.error.details);
      break;
  }
  if (String(payload.type).startsWith('response.'))
    valid = valid && typeof payload.response_id === 'string' && payload.response_id.length > 0;
  if (!valid) throw new Error('MFQ Server returned an invalid response payload');
  return value as ResponseFrame;
}
