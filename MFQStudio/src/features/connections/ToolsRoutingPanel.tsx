/** 独立加载和维护 MCP 工具服务器及远程节点，供资源和连接页面复用。 */
import { useEffect, useState, type FormEvent } from 'react';
import { connectionsApi } from '../../shared/api/resources/connections';
import type { McpServerResource, McpToolResource, RemoteNode } from '../../shared/api/types';
import { SectionLabel, TMPanel } from '../../app/display';
import { errorMessage, formatNumber } from '../../app/formatters';
import { useRuntime } from '../../app/RuntimeProvider';
import { useSettings } from '../settings/SettingsProvider';
import { toast } from '../../stores/toastStore';

/** 页面挂载后读取连接资源，所有写入错误仅影响当前连接面板。 */
export function ToolsRoutingPanel() {
  const { tr } = useSettings();
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

  /** 执行连接写操作并重新获取工具清单，供聊天页下次进入时读取。 */
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
  /** 校验并注册用户填写的 MCP 服务。 */
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
  /** 校验远程服务地址并注册节点。 */
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
        title={tr('工具与路由', 'Tools and routing')}
        subtitle={tr('可选的 MCP 与远程节点', 'Optional MCP and remote nodes')}
      />
      <div className="dashboard-grid server-tools-grid">
        <TMPanel className="mcp-panel">
          <div className="panel-heading">
            <div>
              <h2>MCP</h2>
              <p>{tr('工具服务器与模型可见工具', 'Tool servers and model-visible tools')}</p>
            </div>
            <b>{tr(`${tools.length} 个工具`, `${tools.length} tools`)}</b>
          </div>
          <form className="mcp-form" onSubmit={createMcpServer}>
            <input
              aria-label={tr('服务器名称', 'Server name')}
              onChange={(event) =>
                setMcpDraft((current) => ({ ...current, name: event.target.value }))
              }
              placeholder={tr('名称', 'Name')}
              value={mcpDraft.name}
            />
            <select
              aria-label={tr('传输方式', 'Transport')}
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
                  : tr('可执行文件', 'Executable')
              }
              onChange={(event) =>
                setMcpDraft((current) => ({ ...current, endpoint: event.target.value }))
              }
              placeholder={
                mcpDraft.transport === 'streamable_http'
                  ? 'https://host/mcp'
                  : tr('可执行文件路径', 'Executable path')
              }
              type={mcpDraft.transport === 'streamable_http' ? 'url' : 'text'}
              value={mcpDraft.endpoint}
            />
            <button
              className="primary"
              disabled={!ready || busy || !mcpDraft.name.trim() || !mcpDraft.endpoint.trim()}
              type="submit"
            >
              {tr('添加', 'Add')}
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
                  {server.enabled ? tr('停用', 'Disable') : tr('启用', 'Enable')}
                </button>
                <button
                  disabled={busy}
                  onClick={() => void mutate(() => connectionsApi.deleteMcpServer(server.id))}
                  type="button"
                >
                  {tr('删除', 'Delete')}
                </button>
              </div>
            ))}
          </div>
        </TMPanel>
        <TMPanel className="cluster-panel">
          <div className="panel-heading">
            <div>
              <h2>{tr('远程节点', 'Remote nodes')}</h2>
              <p>
                {tr(
                  '按模型和负载路由到健康的 MFQ Server',
                  'Route by model and load across healthy MFQ Server nodes',
                )}
              </p>
            </div>
            <b>
              {nodes.filter((node) => node.healthy).length} / {nodes.length}
            </b>
          </div>
          <form className="node-form" onSubmit={registerRemoteNode}>
            <input
              aria-label={tr('节点名称', 'Node name')}
              onChange={(event) =>
                setNodeDraft((current) => ({ ...current, name: event.target.value }))
              }
              placeholder={tr('节点名称', 'Node name')}
              value={nodeDraft.name}
            />
            <input
              aria-label={tr('节点地址', 'Node URL')}
              onChange={(event) =>
                setNodeDraft((current) => ({ ...current, url: event.target.value }))
              }
              placeholder="https://worker.example"
              type="url"
              value={nodeDraft.url}
            />
            <input
              aria-label={tr('密钥环境变量', 'Credential environment variable')}
              onChange={(event) =>
                setNodeDraft((current) => ({ ...current, api_key_env: event.target.value }))
              }
              placeholder={tr('密钥环境变量（可选）', 'Credential environment variable (optional)')}
              value={nodeDraft.api_key_env}
            />
            <button disabled={!ready || busy} type="submit">
              {tr('添加', 'Add')}
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
                    {tr(`${node.models.length} 个模型`, `${node.models.length} models`)} ·{' '}
                    {tr(`${node.active_requests} 个活动请求`, `${node.active_requests} active`)}
                    {typeof node.metrics.total_requests === 'number'
                      ? tr(
                          ` · ${formatNumber(node.metrics.total_requests)} 个请求`,
                          ` · ${formatNumber(node.metrics.total_requests)} requests`,
                        )
                      : ''}
                    {node.error ? ` · ${node.error}` : ''}
                  </small>
                </div>
                <button
                  disabled={busy}
                  onClick={() => void mutate(() => connectionsApi.deleteRemoteNode(node.id))}
                  type="button"
                >
                  {tr('删除', 'Delete')}
                </button>
              </div>
            ))}
          </div>
        </TMPanel>
      </div>
    </>
  );
}
