/** 定义 connections 领域的服务契约，仅包含类型，不依赖运行时代码。 */

export interface RemoteNode {
  id: string;
  name: string;
  url: string;
  api_key_env?: string | null;
  enabled: boolean;
  healthy: boolean;
  models: string[];
  active_requests: number;
  metrics: Record<string, unknown>;
  last_checked_at?: string | null;
  error?: string | null;
  created_at: string;
  updated_at: string;
}

export interface McpServerResource {
  id: string;
  name: string;
  transport: 'stdio' | 'streamable_http';
  enabled: boolean;
  url?: string | null;
  command?: string | null;
  args: string[];
  header_env: Record<string, string>;
  timeout_seconds: number;
  created_at: string;
  updated_at: string;
}

export interface McpToolResource {
  server_id: string;
  server: string;
  name: string;
  qualified_name: string;
  description?: string | null;
  input_schema: Record<string, unknown>;
}

export interface McpToolCallResult {
  server: string;
  name: string;
  content: Array<Record<string, unknown>>;
  structured_content?: Record<string, unknown> | null;
  is_error: boolean;
}
