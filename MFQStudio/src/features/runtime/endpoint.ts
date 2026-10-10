/** OpenAI SDK base URL; keep the Studio management API URL unchanged. */
export function openAIEndpoint(serviceUrl?: string | null): string {
  const url = new URL(serviceUrl || 'http://127.0.0.1:8090');
  const path = url.pathname.replace(/\/+$/, '');
  url.pathname = path.endsWith('/v1') ? path : `${path}/v1`;
  url.search = '';
  url.hash = '';
  return url.toString().replace(/\/$/, '');
}

export function anthropicEndpoint(serviceUrl: string | null | undefined, port: number | null | undefined): string | null {
  if (port == null || !Number.isInteger(port) || port < 1 || port > 65535) return null;
  const url = new URL(serviceUrl || 'http://127.0.0.1:8090');
  url.port = String(port);
  const path = url.pathname.replace(/\/+$/, '').replace(/\/v1$/, '');
  url.pathname = `${path}/v1/messages`;
  url.search = '';
  url.hash = '';
  return url.toString().replace(/\/$/, '');
}
