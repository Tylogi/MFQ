/** 解析 Hugging Face 与 ModelScope 模型仓库引用及版本。 */
import { HubModelSummary } from '../../shared/api/types';

export interface HubReference {
  provider: HubModelSummary["provider"];
  repoId: string;
  revision?: string;
}

/** 解析模型仓库名称或链接，返回提供方、仓库及可选版本。 */
export function parseHubReference(
  value: string,
  fallbackProvider: HubModelSummary["provider"],
): HubReference | null {
  const trimmed = value.trim();
  if (!trimmed) return null;
  if (/^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(trimmed)) {
    return { provider: fallbackProvider, repoId: trimmed };
  }
  let parsed: URL;
  try {
    parsed = new URL(trimmed);
  } catch {
    return null;
  }
  const host = parsed.hostname.toLowerCase();
  const provider = host === "huggingface.co" || host === "www.huggingface.co"
    ? "huggingface"
    : host === "modelscope.cn" || host === "www.modelscope.cn"
      ? "modelscope"
      : null;
  if (!provider) return null;
  let parts: string[];
  try {
    parts = parsed.pathname.split("/").filter(Boolean).map(decodeURIComponent);
  } catch {
    return null;
  }
  if (provider === "modelscope" && parts[0] === "models") parts.shift();
  if (parts.length < 2) return null;
  const marker = parts.findIndex((part) => part === "tree");
  const repoId = parts.slice(0, 2).join("/");
  const revision = marker >= 0 && parts.length > marker + 1
    ? parts.slice(marker + 1).join("/")
    : undefined;
  return { provider, repoId, revision };
}
