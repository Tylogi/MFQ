/** OpenAI SDK base URL; keep the Studio management API URL unchanged. */
export function openAIEndpoint(serviceUrl?: string | null): string {
  const url = new URL(serviceUrl || 'http://127.0.0.1:8090');
  const path = url.pathname.replace(/\/+$/, '');
  url.pathname = path.endsWith('/v1') ? path : `${path}/v1`;
  url.search = '';
  url.hash = '';
  return url.toString().replace(/\/$/, '');
}
