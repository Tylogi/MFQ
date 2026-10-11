import { useEffect, useState, type FormEvent } from 'react';
import { connectionsApi } from '../../shared/api/resources/connections';
import type { RemoteNode } from '../../shared/api/types';
import { SectionLabel, TMPanel } from '../../app/display';
import { errorMessage, formatNumber } from '../../app/formatters';
import { useRuntime } from '../../app/RuntimeProvider';
import { useConnectionScope } from '../../app/useConnectionScope';
import { useSettings } from '../settings/SettingsProvider';
import { toast } from '../../stores/toastStore';

export function RemoteRoutingPanel() {
  const { tr } = useSettings();
  const { ready, connectionRevision } = useRuntime();
  const connectionScope = useConnectionScope();
  const [nodes, setNodes] = useState<RemoteNode[]>([]);
  const [nodeDraft, setNodeDraft] = useState({ name: '', url: '', api_key_env: '' });
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    if (!ready) return;
    let disposed = false;
    void connectionsApi.remoteNodes(true)
      .then(nodes => {
        if (!disposed) setNodes(nodes);
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

  async function mutate(operation: (current: () => boolean) => Promise<unknown>) {
    const current = connectionScope();
    if (busy) return;
    setBusy(true);
    try {
      await operation(current);
      if (!current()) return;
      const nextNodes = await connectionsApi.remoteNodes(true);
      if (!current()) return;
      setNodes(nextNodes);
    } catch (cause) {
      if (current()) toast.error(errorMessage(cause));
    } finally {
      if (current()) setBusy(false);
    }
  }
  function registerRemoteNode(event: FormEvent) {
    event.preventDefault();
    if (!nodeDraft.name.trim() || !nodeDraft.url.trim()) return;
    void mutate(async (current) => {
      await connectionsApi.createRemoteNode({
        name: nodeDraft.name.trim(),
        url: nodeDraft.url.trim(),
        api_key_env: nodeDraft.api_key_env.trim() || null,
        enabled: true,
      });
      if (current()) setNodeDraft({ name: '', url: '', api_key_env: '' });
    });
  }
  return (
    <>
      <SectionLabel
        title={tr('远程路由', 'Remote routing')}
        subtitle={tr('可选的远程推理节点', 'Optional remote inference nodes')}
      />
      <div className="dashboard-grid remote-routing-grid">
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
