/** Independently load and manage MCP tool servers and remote nodes for reuse by resources and connections pages. */
import { useTranslation } from 'react-i18next';
import { useEffect, useState, type FormEvent } from 'react';
import { connectionsApi } from '../../shared/api/resources/connections';
import type { McpServerResource, McpToolResource, RemoteNode } from '../../shared/api/types';
import { SectionLabel, TMPanel } from '../../app/display';
import { errorMessage, formatNumber } from '../../app/formatters';
import { useRuntime } from '../../app/RuntimeProvider';
import { toast } from '../../stores/toastStore';
/** Load connection resources on mount; write errors affect only the current connections panel. */
export function ToolsRoutingPanel() {
  const { t } = useTranslation();
  const { ready, connectionRevision } = useRuntime();
  const [servers, setServers] = useState<McpServerResource[]>([]);
  const [tools, setTools] = useState<McpToolResource[]>([]);
  const [nodes, setNodes] = useState<RemoteNode[]>([]);
  const [mcpDraft, setMcpDraft] = useState({
    name: '',
    transport: 'streamable_http' as 'stdio' | 'streamable_http',
    endpoint: '',
  });
  const [nodeDraft, setNodeDraft] = useState({ name: '', url: '', api_key_env: '' });
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    if (!ready) return;
    let disposed = false;
    void Promise.all([connectionsApi.mcpServers(), connectionsApi.mcpTools(), connectionsApi.remoteNodes(true)])
      .then(([servers, tools, nodes]) => {
        if (!disposed) {
          setServers(servers);
          setTools(tools.data);
          setNodes(nodes);
        }
      })
      .catch((cause) => {
        if (!disposed) {
          toast.error(errorMessage(cause));
        }
      });
    return () => {
      disposed = true;
    };
  }, [ready, connectionRevision]);
/** Perform a connection write and reload the tool catalog for the next chat-page visit. */
  async function mutate(operation: () => Promise<unknown>) {
    if (busy) return;
    setBusy(true);
    try {
      await operation();
      const [nextServers, nextTools, nextNodes] = await Promise.all([
        connectionsApi.mcpServers(),
        connectionsApi.mcpTools(),
        connectionsApi.remoteNodes(true),
      ]);
      setServers(nextServers);
      setTools(nextTools.data);
      setNodes(nextNodes);
      window.dispatchEvent(new Event('mfq:tools-changed'));
    } catch (cause) {
      toast.error(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }
/** Validate and register the MCP service entered by the user. */
  function createMcpServer(event: FormEvent) {
    event.preventDefault();
    if (!mcpDraft.name.trim() || !mcpDraft.endpoint.trim()) return;
    void mutate(async () => {
      await connectionsApi.createMcpServer({
        name: mcpDraft.name.trim(),
        transport: mcpDraft.transport,
        enabled: true,
        url: mcpDraft.transport === 'streamable_http' ? mcpDraft.endpoint.trim() : null,
        command: mcpDraft.transport === 'stdio' ? mcpDraft.endpoint.trim() : null,
      });
      setMcpDraft({ name: '', transport: 'streamable_http', endpoint: '' });
    });
  }
/** Validate the remote service URL and register the node. */
  function registerRemoteNode(event: FormEvent) {
    event.preventDefault();
    if (!nodeDraft.name.trim() || !nodeDraft.url.trim()) return;
    void mutate(async () => {
      await connectionsApi.createRemoteNode({
        name: nodeDraft.name.trim(),
        url: nodeDraft.url.trim(),
        api_key_env: nodeDraft.api_key_env.trim() || null,
        enabled: true,
      });
      setNodeDraft({ name: '', url: '', api_key_env: '' });
    });
  }
  return (
    <>
      <SectionLabel
        title={t('connections:toolsRoutingPanel.toolsAndRouting')}
        subtitle={t('connections:toolsRoutingPanel.optionalMcpAndRemoteNodes')}
      />
      <div className="dashboard-grid server-tools-grid">
        <TMPanel className="mcp-panel">
          <div className="panel-heading">
            <div>
              <h2>MCP</h2>
              <p>{t('connections:toolsRoutingPanel.toolServersAndModelVisibleTools')}</p>
            </div>
            <b>{t('connections:toolsRoutingPanel.tools', { count: tools.length })}</b>
          </div>
          <form className="mcp-form" onSubmit={createMcpServer}>
            <input
              aria-label={t('connections:toolsRoutingPanel.serverName')}
              onChange={(event) =>
                setMcpDraft((current) => ({ ...current, name: event.target.value }))
              }
              placeholder={t('connections:toolsRoutingPanel.name')}
              value={mcpDraft.name}
            />
            <select
              aria-label={t('connections:toolsRoutingPanel.transport')}
              onChange={(event) =>
                setMcpDraft((current) => ({
                  ...current,
                  transport: event.target.value as 'stdio' | 'streamable_http',
                }))
              }
              value={mcpDraft.transport}
            >
              <option value="streamable_http">HTTP</option>
              <option value="stdio">stdio</option>
            </select>
            <input
              aria-label={
                mcpDraft.transport === 'streamable_http'
                  ? 'Streamable HTTP URL'
                  : t('connections:toolsRoutingPanel.executable')
              }
              onChange={(event) =>
                setMcpDraft((current) => ({ ...current, endpoint: event.target.value }))
              }
              placeholder={
                mcpDraft.transport === 'streamable_http'
                  ? 'https://host/mcp'
                  : t('connections:toolsRoutingPanel.executablePath')
              }
              type={mcpDraft.transport === 'streamable_http' ? 'url' : 'text'}
              value={mcpDraft.endpoint}
            />
            <button
              className="primary"
              disabled={!ready || busy || !mcpDraft.name.trim() || !mcpDraft.endpoint.trim()}
              type="submit"
            >
              {t('connections:toolsRoutingPanel.add')}
            </button>
          </form>
          <div className="mcp-server-list">
            {servers.map((server) => (
              <div className="mcp-server" key={server.id}>
                <span className={server.enabled ? 'model-state active' : 'model-state'} />
                <div>
                  <strong>{server.name}</strong>
                  <small>
                    {server.transport} · {server.url || server.command}
                  </small>
                </div>
                <button
                  disabled={busy}
                  onClick={() => void mutate(() => connectionsApi.updateMcpServer(server.id, !server.enabled))}
                  type="button"
                >
                  {server.enabled ? t('connections:toolsRoutingPanel.disable') : t('connections:toolsRoutingPanel.enable')}
                </button>
                <button
                  disabled={busy}
                  onClick={() => void mutate(() => connectionsApi.deleteMcpServer(server.id))}
                  type="button"
                >
                  {t('common:delete')}
                </button>
              </div>
            ))}
          </div>
        </TMPanel>
        <TMPanel className="cluster-panel">
          <div className="panel-heading">
            <div>
              <h2>{t('connections:toolsRoutingPanel.remoteNodes')}</h2>
              <p>
                {t('connections:toolsRoutingPanel.routeByModelAndLoadAcrossHealthyMfqServerNodes')}
              </p>
            </div>
            <b>
              {nodes.filter((node) => node.healthy).length} / {nodes.length}
            </b>
          </div>
          <form className="node-form" onSubmit={registerRemoteNode}>
            <input
              aria-label={t('connections:toolsRoutingPanel.nodeName')}
              onChange={(event) =>
                setNodeDraft((current) => ({ ...current, name: event.target.value }))
              }
              placeholder={t('connections:toolsRoutingPanel.nodeName')}
              value={nodeDraft.name}
            />
            <input
              aria-label={t('connections:toolsRoutingPanel.nodeUrl')}
              onChange={(event) =>
                setNodeDraft((current) => ({ ...current, url: event.target.value }))
              }
              placeholder="https://worker.example"
              type="url"
              value={nodeDraft.url}
            />
            <input
              aria-label={t('connections:toolsRoutingPanel.credentialEnvironmentVariable')}
              onChange={(event) =>
                setNodeDraft((current) => ({ ...current, api_key_env: event.target.value }))
              }
              placeholder={t('connections:toolsRoutingPanel.credentialEnvironmentVariableOptional')}
              value={nodeDraft.api_key_env}
            />
            <button disabled={!ready || busy} type="submit">
              {t('connections:toolsRoutingPanel.add')}
            </button>
          </form>
          <div className="node-list">
            {nodes.map((node) => (
              <div key={node.id}>
                <span className={node.healthy ? 'model-state active' : 'model-state failed'} />
                <div>
                  <strong>{node.name}</strong>
                  <small>
                    {node.url} ·{' '}
                    {t('connections:toolsRoutingPanel.models', { count: node.models.length })} ·{' '}
                    {t('connections:toolsRoutingPanel.active', { activeRequests: node.active_requests })}
                    {typeof node.metrics.total_requests === 'number'
                      ? t('connections:toolsRoutingPanel.requests', { requests: formatNumber(node.metrics.total_requests) })
                      : ''}
                    {node.error ? ` · ${node.error}` : ''}
                  </small>
                </div>
                <button
                  disabled={busy}
                  onClick={() => void mutate(() => connectionsApi.deleteRemoteNode(node.id))}
                  type="button"
                >
                  {t('common:delete')}
                </button>
              </div>
            ))}
          </div>
        </TMPanel>
      </div>
    </>
  );
}
