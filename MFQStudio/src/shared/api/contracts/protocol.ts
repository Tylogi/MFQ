/** Define service contracts for the protocol domain using types only, with no runtime dependencies. */

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
