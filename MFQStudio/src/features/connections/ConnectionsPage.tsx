/** Configure desktop and browser service connections using the actual connected service address. */
import { useEffect, useState } from 'react';
import { Icon, ScreenHeader, SectionLabel, SettingRow, TMPanel } from '../../app/display';
import { errorMessage } from '../../app/formatters';
import {
  configureStudio,
  isStudio,
  saveStudioCredential,
  studioCredential,
  type StudioConfig,
} from '../../studio';
import { useRuntime } from '../../app/RuntimeProvider';
import { runtimeModelNames } from '../runtime/modelSelection';
import { runtimeApi } from '../../shared/api/resources/runtime';
import { resolveServiceUrl, setApiToken, setBrowserServiceUrl } from '../../shared/api/client';
import { useSettings } from '../settings/SettingsProvider';
import { ToolsRoutingPanel } from './ToolsRoutingPanel';
import { MemorySettingsPanel } from './MemorySettingsPanel';
import { RuntimeProfilesPanel } from '../runtime/RuntimeProfilesPanel';
import { ModelAliasMapping } from './ModelAliasMapping';
import { toast } from '../../stores/toastStore';

function browserConfig(): StudioConfig {
  const address = resolveServiceUrl();
  const url = new URL(address);
  return {
    mode: ['127.0.0.1', 'localhost', '[::1]'].includes(url.hostname) ? 'local' : 'remote',
    remote_url: address, local_service_port: Number(url.port) || (url.protocol === 'https:' ? 443 : 80),
  };
}

/** Display and save service connection settings, reconnecting only after a successful update. */
export function ConnectionsPage() {
  const { tr } = useSettings();
  const {
    runtime,
    models,
    instances,
    selectedModel,
    studio,
    reloadService,
  } = useRuntime();
  const [draft, setDraft] = useState<StudioConfig>(() => studio?.config ?? browserConfig());
  const [listeningPort, setListeningPort] = useState<number | null>(null);
  const [token, setToken] = useState('');
  const [credentialWritable, setCredentialWritable] = useState(false);
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    if (studio) setDraft(studio.config);
    else {
      let disposed = false;
      void runtimeApi.runtimeListener().then((listener) => {
        if (!disposed) {
          setDraft((current) => ({ ...current, local_service_port: listener.port }));
          setListeningPort(listener.port);
        }
      }).catch(() => {});
      return () => { disposed = true; };
    }
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
  const active = Boolean(studio?.reachable ?? runtime);
  const modelNames = runtimeModelNames(models, instances);


  async function save() {
    if (!draft || busy) return;
    setBusy(true);
    try {
      if (!Number.isInteger(draft.local_service_port) || draft.local_service_port < 1 || draft.local_service_port > 65535) {
        throw new Error(tr('端口必须为 1–65535 的整数', 'Port must be an integer between 1 and 65535'));
      }
      if (isStudio()) {
        if (studio?.config.mode === 'local' && draft.mode === 'local'
            && studio.config.local_service_port !== draft.local_service_port) {
          await runtimeApi.configureRuntimeListener(draft.local_service_port);
        }
        await configureStudio(draft);
        if (credentialWritable) await saveStudioCredential(token);
      } else {
        let address = draft.remote_url.trim().replace(/\/+$/, '').replace(/\/v1$/, '');
        if (draft.mode === 'local') {
          const current = browserConfig();
          if (current.mode === 'local') await runtimeApi.configureRuntimeListener(draft.local_service_port);
          const local = new URL(current.mode === 'local' ? current.remote_url : 'http://127.0.0.1');
          local.port = String(draft.local_service_port);
          address = local.toString().replace(/\/+$/, '');
        } else {
          const parsed = new URL(address);
          if (!['http:', 'https:'].includes(parsed.protocol) || parsed.username || parsed.password) {
            throw new Error(tr('请输入不含凭据的 HTTP 或 HTTPS 服务地址', 'Enter an HTTP or HTTPS service URL without credentials'));
          }
        }
        setBrowserServiceUrl(address);
        if (credentialWritable) setApiToken(token);
        if (draft.mode === 'local' && window.location.port === String(listeningPort)
            && listeningPort !== draft.local_service_port) {
          const page = new URL(window.location.href);
          page.port = String(draft.local_service_port);
          window.location.assign(page.toString());
          return;
        }
      }
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
        title={tr('服务', 'Service')}
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
              title={tr('模型 ID', 'Model ID')}
              detail={tr(
                '由 /v1/models 公布，并用于对话补全请求。',
                'Advertised by /v1/models and accepted by chat completions.',
              )}
              trailing={
                <div className="server-row-actions server-model-control">
                  <ModelAliasMapping models={modelNames} selectedModel={selectedModel} />
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
                    isStudio() ? '凭据只保存在系统凭据库中。' : '凭据仅保留在当前页面内存中。',
                    isStudio() ? 'The credential is stored only in the system credential vault.' : 'The credential stays only in this page’s memory.',
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
      <RuntimeProfilesPanel />
      <ToolsRoutingPanel />
    </section>
  );
}
