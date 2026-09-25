/** 定义 protocol 领域的服务契约，仅包含类型，不依赖运行时代码。 */

export interface ApiErrorBody {
  error: {
    code: string;
    message: string;
    retryable: boolean;
    details: Record<string, unknown>;
  };
}

export interface RealtimeFrame {
  protocol_version: '1.0';
  session_id: string;
  sequence: number;
  timestamp: string;
  payload: Record<string, unknown> & { type: string };
}
