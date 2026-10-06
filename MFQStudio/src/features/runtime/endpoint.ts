/** OpenAI SDK base URL; keep the Studio management API URL unchanged. */
import { resolveServiceUrl } from '../../shared/api/client';

/** Resolve an SDK endpoint from the active service, including browser same-origin connections. */
export function openAIEndpoint(serviceUrl?: string | null): string {
  const url = new URL(resolveServiceUrl(serviceUrl ?? undefined));
  const path = url.pathname.replace(/\/+$/, '');
  url.pathname = path.endsWith('/v1') ? path : `${path}/v1`;
  url.search = '';
  url.hash = '';
  return url.toString().replace(/\/$/, '');
}
