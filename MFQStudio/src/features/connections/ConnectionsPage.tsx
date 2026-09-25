/** 连接页面负责服务器配置草稿、凭据保存及服务重连，业务状态不流入应用外壳。 */
import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router';
import { connectionsApi } from '../../shared/api/resources/connections';
import { Icon, ScreenHeader, SectionLabel, SettingRow, TMPanel } from '../../app/display';
import { errorMessage, formatBytes, formatNumber } from '../../app/formatters';
import { STUDIO_PATHS } from '../../navigation';
import {
  configureStudio,
  saveStudioCredential,
  studioConfirm,
  studioCredential,
  type StudioConfig,
} from '../../studio';
import { useRuntime } from '../../app/RuntimeProvider';
import { runtimeModelNames } from '../runtime/modelSelection';
import { modeTemplateSettings, type GenerationSettings } from '../settings/configuration';
import { useSettings } from '../settings/SettingsProvider';
import { ToolsRoutingPanel } from './ToolsRoutingPanel';
import { MemorySettingsPanel } from './MemorySettingsPanel';
import { toast } from '../../stores/toastStore';
import { InferenceDefaultsPanel } from './InferenceDefaultsPanel';

/** 提供运行配置、内存与缓存概览，以及连接页自己的保存和重载操作。 */
export function ConnectionsPage() {
  const { settings, replaceSettings, tr, contextSize, setContextSize } = useSettings();
  const {
    runtime,
    realtime,
    models,
    instances,
    selectedModel,
    setSelectedModel,
    studio,
    reloadService,
    refreshRuntime,
  } = useRuntime();
  const navigate = useNavigate();
  const [draft, setDraft] = useState<StudioConfig | null>(studio?.config ?? null);
  const [token, setToken] = useState('');
  const [credentialWritable, setCredentialWritable] = useState(false);
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    setDraft(studio?.config ?? null);
  }, [studio]);
  useEffect(() => {
    let disposed = false;
    void studioCredential()
      .then((value) => {
        if (!disposed) setToken(value ?? '');
      })
      .catch((cause) => {
        if (!disposed) {
          toast.error(errorMessage(cause));
        }
      });
    return () => {
      disposed = true;
    };
  }, []);
  const active = Boolean(studio?.reachable);
  const modelNames = runtimeModelNames(models, instances);


  /** 将局部服务草稿提交平台并触发应用级连接版本更新。 */
  async function save() {
    if (!draft || busy) return;
    setBusy(true);
    try {
      await configureStudio(draft);
      if (credentialWritable) await saveStudioCredential(token);
      const reconnected = await reloadService();
      setCredentialWritable(false);
      if (reconnected) toast.success(tr('服务器设置已保存', 'Server settings saved'));
    } catch (cause) {
      toast.error(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }


  return (
    <section className="dashboard-view">
      <ScreenHeader
        title={tr('服务器', 'Server')}
        subtitle={tr(
          '运行服务、连接与模型默认值。',
          'Runtime service, connections, and model defaults.',
        )}
      />
      <div className="server-page">
        {active && (
          <div className="server-active-notice">
            <Icon name="info" size={15} />
            <span>
              {tr(
                '服务器正在运行；网络设置保存后会立即重新连接。',
                'The server is active. Network changes reconnect as soon as they are saved.',
              )}
            </span>
          </div>
        )}
        <SectionLabel title={tr('运行服务', 'Runtime')} />
        <TMPanel className="server-settings-panel">
          <div className="setting-list">
            <SettingRow
              title={tr('Runtime 可执行文件', 'Runtime executable')}
              detail={tr(
                '应用已包含推理服务，并自动使用本机 Metal Runtime。',
                'The packaged app includes the inference server and discovers the local Metal runtime automatically.',
              )}
              trailing={
                <div className="server-row-actions">
                  <code>mfq-cli → mfq-decode-metal</code>
                  <span className={`runtime-status-pill ${active ? 'running' : 'stopped'}`}>
                    <i />
                    {active ? tr('已连接', 'Connected') : tr('离线', 'Offline')}
                  </span>
                </div>
              }
            />
            <SettingRow
              title={tr('模型 ID', 'Model ID')}
              detail={tr(
                '由 /v1/models 公布，并用于对话补全请求。',
                'Advertised by /v1/models and accepted by chat completions.',
              )}
              trailing={
                <div className="server-row-actions server-model-control">
                  {modelNames.length > 1 ? (
                    <select
                      aria-label={tr('当前模型', 'Current model')}
                      disabled={busy}
                      onChange={(event) => setSelectedModel(event.target.value)}
                      value={selectedModel}
                    >
                      {modelNames.map((name) => (
                        <option key={name} value={name}>
                          {name}
                        </option>
                      ))}
                    </select>
                  ) : (
                    <strong title={selectedModel}>
                      {selectedModel || tr('尚未加载', 'Not loaded')}
                    </strong>
                  )}
                  <button onClick={() => navigate(STUDIO_PATHS.models)} type="button">
                    {tr('选择…', 'Choose…')}
                  </button>
                </div>
              }
            />
            <SettingRow
              title={tr('绑定地址', 'Bind address')}
              detail={tr(
                '本地模式仅监听 127.0.0.1；远程模式连接另一台 MFQ Server。',
                'Local mode stays on 127.0.0.1; remote mode connects to another MFQ Server.',
              )}
              trailing={
                <select
                  aria-label={tr('绑定地址', 'Bind address')}
                  disabled={busy || !draft}
                  onChange={(event) =>
                    setDraft(
                      (current) =>
                        current && { ...current, mode: event.target.value as StudioConfig['mode'] },
                    )
                  }
                  value={draft?.mode ?? 'local'}
                >
                  <option value="local">
                    {tr('仅本机 · 127.0.0.1', 'Local only · 127.0.0.1')}
                  </option>
                  <option value="remote">{tr('远程 MFQ Server', 'Remote MFQ Server')}</option>
                </select>
              }
            />
            {draft?.mode === 'remote' ? (
              <>
                <SettingRow
                  title={tr('远程端点', 'Remote endpoint')}
                  detail={tr(
                    '远程 MFQ Server 的 OpenAI 兼容基础 URL。',
                    'OpenAI-compatible base URL for the remote MFQ Server.',
                  )}
                  trailing={
                    <input
                      aria-label={tr('远程端点', 'Remote endpoint')}
                      className="server-wide-input"
                      disabled={busy}
                      onChange={(event) =>
                        setDraft(
                          (current) => current && { ...current, remote_url: event.target.value },
                        )
                      }
                      placeholder="https://host:port"
                      type="url"
                      value={draft.remote_url}
                    />
                  }
                />
                <SettingRow
                  title={tr('API 密钥', 'API key')}
                  detail={tr(
                    '凭据只保存在系统凭据库中。',
                    'The credential is stored only in the system credential vault.',
                  )}
                  trailing={
                    <input
                      aria-label={tr('API 密钥', 'API key')}
                      autoComplete="off"
                      className="server-wide-input"
                      disabled={busy}
                      onChange={(event) => {
                        setToken(event.target.value);
                        setCredentialWritable(true);
                      }}
                      placeholder={tr('可选', 'Optional')}
                      type="password"
                      value={token}
                    />
                  }
                />
              </>
            ) : (
              <SettingRow
                title={tr('端口', 'Port')}
                detail={tr(
                  'OpenAI 兼容 HTTP 服务使用的 TCP 端口。',
                  'TCP port used by the OpenAI-compatible HTTP server.',
                )}
                trailing={
                  <input
                    aria-label={tr('端口', 'Port')}
                    className="server-number-input"
                    disabled={busy || !draft}
                    max={65535}
                    min={1}
                    onChange={(event) =>
                      setDraft(
                        (current) =>
                          current && { ...current, local_service_port: Number(event.target.value) },
                      )
                    }
                    type="number"
                    value={draft?.local_service_port ?? 8090}
                  />
                }
              />
            )}
          </div>
        </TMPanel>
        <MemorySettingsPanel />
        <InferenceDefaultsPanel />
        <SectionLabel title={tr('自动化', 'Automation')} />
        <TMPanel className="server-settings-panel">
          <div className="setting-list">
            <SettingRow
              title={tr('MFQ Studio 打开时启动服务器', 'Start server when MFQ Studio opens')}
              detail={tr(
                '本地模式会自动恢复服务，并使用当前模型和已保存的运行配置。',
                'Local mode restores the server automatically with the current model and saved runtime configuration.',
              )}
              trailing={
                <input
                  aria-label={tr(
                    'MFQ Studio 打开时启动服务器',
                    'Start server when MFQ Studio opens',
                  )}
                  checked={draft?.mode !== 'remote'}
                  disabled
                  readOnly
                  type="checkbox"
                />
              }
            />
          </div>
        </TMPanel>
        <div className="server-page-footer">
          <span>{tr('对话默认值会自动保存。', 'Chat defaults are saved automatically.')}</span>
          <button
            className="primary"
            disabled={busy || !draft}
            onClick={() => void save()}
            type="button"
          >
            {tr('保存服务器设置', 'Save server settings')}
          </button>
        </div>
      </div>
      <ToolsRoutingPanel />
    </section>
  );
}
