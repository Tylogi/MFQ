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
