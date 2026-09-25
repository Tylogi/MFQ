/** 定义 jobs 领域的服务契约，仅包含类型，不依赖运行时代码。 */
import type { RuntimeLogEntry } from './runtime';
import type { ApiErrorBody } from './protocol';

export interface JobResource {
  id: string;
  kind: string;
  status:
    'queued' | 'running' | 'cancelling' | 'succeeded' | 'failed' | 'cancelled' | 'interrupted';
  payload: Record<string, unknown>;
  progress: number;
  cancel_requested: boolean;
  result?: Record<string, unknown> | null;
  error?: ApiErrorBody['error'] | null;
  created_at: string;
  updated_at: string;
}

export interface JobEventResource {
  job_id: string;
  sequence: number;
  type: 'state' | 'progress' | 'log' | 'artifact';
  level: RuntimeLogEntry['level'];
  message?: string | null;
  progress?: number | null;
  data: Record<string, unknown>;
  created_at: string;
}

export interface JsonSchemaProperty {
  type?: string | string[];
  title?: string;
  description?: string;
  default?: unknown;
  enum?: unknown[];
  minimum?: number;
  maximum?: number;
  anyOf?: JsonSchemaProperty[];
  items?: JsonSchemaProperty;
}

export interface JobKindResource {
  kind: string;
  payload_schema: {
    type?: string;
    properties?: Record<string, JsonSchemaProperty>;
    required?: string[];
    [key: string]: unknown;
  };
}
