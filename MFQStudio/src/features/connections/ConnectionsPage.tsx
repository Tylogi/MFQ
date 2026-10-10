import { useEffect, useRef, useState } from 'react';
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
import { getApiBaseUrl, getApiToken, setApiToken, setBrowserServiceUrl } from '../../shared/api/client';
import { useSettings } from '../settings/SettingsProvider';
import { RemoteRoutingPanel } from './RemoteRoutingPanel';
import { MemorySettingsPanel } from './MemorySettingsPanel';
import { ContextManagementPanel } from './ContextManagementPanel';
import { PrefixCacheSettingsPanel } from './PrefixCacheSettingsPanel';
import { RuntimeProfilesPanel } from '../runtime/RuntimeProfilesPanel';
import { ModelAliasMapping } from './ModelAliasMapping';
import { InferencePolicyPanel } from './InferencePolicyPanel';
import { toast } from '../../stores/toastStore';

function browserConfig(): StudioConfig {
  const address = getApiBaseUrl() || 'http://127.0.0.1:8090';
  const url = new URL(address);
  return {
    mode: ['127.0.0.1', 'localhost', '[::1]'].includes(url.hostname) ? 'local' : 'remote',
    remote_url: address, local_service_port: Number(url.port) || 8090,
  };
}

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
  const [anthropicPort, setAnthropicPort] = useState<number | null>(null);
  const [anthropicDraft, setAnthropicDraft] = useState('');
  const [configurable, setConfigurable] = useState(false);
  const [token, setToken] = useState('');
  const [credentialWritable, setCredentialWritable] = useState(false);
  const [busy, setBusy] = useState(false);
  const [applyState, setApplyState] = useState<{ error: boolean; text: string } | null>(null);
  const applying = useRef(false);
  const applied = useRef(draft);
  const mounted = useRef(false);
  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; };
  }, []);
  useEffect(() => {
    if (studio) { setDraft(studio.config); applied.current = studio.config; }
    let disposed = false;
    void runtimeApi.runtimeListener().then((listener) => {
      if (!disposed) {
        if (!studio) {
          setDraft((current) => ({ ...current, local_service_port: listener.port }));
          applied.current = { ...applied.current, local_service_port: listener.port };
        }
        setListeningPort(listener.port);
        setConfigurable(listener.configurable);
        setAnthropicPort(listener.anthropic_port ?? null);
        setAnthropicDraft(listener.anthropic_port == null ? '' : String(listener.anthropic_port));
      }
    }).catch(() => {});
    return () => { disposed = true; };
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


  async function apply(draft: StudioConfig) {
    if (applying.current) return;
    if (JSON.stringify(draft) === JSON.stringify(applied.current) && !credentialWritable && !applyState?.error) return;
    const endpoint = getApiBaseUrl();
    let credential = getApiToken();
    const current = () => getApiBaseUrl() === endpoint && getApiToken() === credential;
    applying.current = true;
    setBusy(true);
    setApplyState({ error: false, text: tr('正在应用…', 'Applying…') });
    try {
      if (draft.mode === 'local' && (!Number.isInteger(draft.local_service_port) || draft.local_service_port < 1 || draft.local_service_port > 65535)) {
        throw new Error(tr('端口必须为 1–65535 的整数', 'Port must be an integer between 1 and 65535'));
      }
      if (isStudio()) {
        if (studio?.config.mode === 'local' && draft.mode === 'local'
            && studio.config.local_service_port !== draft.local_service_port) {
          await runtimeApi.configureRuntimeListener(draft.local_service_port);
          if (!current()) return;
        }
        await configureStudio(draft);
        if (!current()) return;
        if (credentialWritable) {
          await saveStudioCredential(token);
          if (!current()) return;
        }
      } else {
        let address = draft.remote_url.trim().replace(/\/+$/, '').replace(/\/v1$/, '');
        if (draft.mode === 'local') {
          if (browserConfig().mode === 'local') await runtimeApi.configureRuntimeListener(draft.local_service_port);
          if (!current()) return;
          address = `http://127.0.0.1:${draft.local_service_port}`;
        } else {
          const parsed = new URL(address);
          if (!['http:', 'https:'].includes(parsed.protocol) || parsed.username || parsed.password) {
            throw new Error(tr('请输入不含凭据的 HTTP 或 HTTPS 服务地址', 'Enter an HTTP or HTTPS service URL without credentials'));
          }
        }
        setBrowserServiceUrl(address);
        if (credentialWritable) { setApiToken(token); credential = token.trim(); }
        if (draft.mode === 'local' && window.location.port === String(listeningPort)
            && listeningPort !== draft.local_service_port) {
          const page = new URL(window.location.href);
          page.port = String(draft.local_service_port);
          window.location.assign(page.toString());
          return;
        }
      }
      if (mounted.current && draft.mode === 'local') setListeningPort(draft.local_service_port);
      const reconnected = await reloadService();
      applied.current = draft;
      if (mounted.current && reconnected) {
        setCredentialWritable(false);
        setApplyState({ error: false, text: tr('已应用，无需重载模型', 'Applied without reloading models') });
      } else if (mounted.current) {
        setApplyState({ error: true, text: tr('设置已应用，但连接失败，请检查地址和凭据', 'Settings applied, but reconnection failed. Check the address and credentials.') });
      }
    } catch (cause) {
      if (mounted.current && current()) {
        setApplyState({ error: true, text: errorMessage(cause) });
      }
    } finally {
      applying.current = false;
      if (mounted.current) setBusy(false);
    }
  }

  async function applyAnthropic() {
    const port = Number(anthropicDraft);
    if (applying.current || port === anthropicPort) return;
    if (!Number.isInteger(port) || port < 1 || port > 65535) {
      setApplyState({ error: true, text: tr('端口必须为 1–65535 的整数', 'Port must be an integer between 1 and 65535') });
      return;
    }
    const endpoint = getApiBaseUrl();
    const credential = getApiToken();
    applying.current = true;
    setBusy(true);
    setApplyState({ error: false, text: tr('正在应用…', 'Applying…') });
    try {
      const listener = await runtimeApi.configureRuntimeListener(port, 'anthropic');
      if (mounted.current && endpoint === getApiBaseUrl() && credential === getApiToken()) {
        setAnthropicPort(listener.anthropic_port ?? port);
        setApplyState({ error: false, text: tr('已应用，无需重载模型', 'Applied without reloading models') });
      }
    } catch (cause) {
      if (mounted.current && endpoint === getApiBaseUrl() && credential === getApiToken()) {
        setApplyState({ error: true, text: errorMessage(cause) });
      }
    } finally {
      applying.current = false;
      if (mounted.current) setBusy(false);
    }
  }


  return (
    <section className="dashboard-view service-view">
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
                '服务器正在运行；网络设置失焦或按回车自动应用，无需重载模型。',
                'The server is active. Network settings apply on blur or Enter without reloading models.',
              )}
            </span>
          </div>
        )}
        <SectionLabel title={tr('API参数', 'API parameters')} />
        <TMPanel className="server-settings-panel server-api-panel">
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
                  onChange={(event) => {
                    const next = { ...draft, mode: event.target.value as StudioConfig['mode'] };
                    setDraft(next);
                    void apply(next);
                  }}
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
                      onBlur={() => void apply(draft)}
                      onKeyDown={(event) => { if (event.key === 'Enter') { event.preventDefault(); void apply(draft); } }}
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
                      onBlur={() => void apply(draft)}
                      onKeyDown={(event) => { if (event.key === 'Enter') { event.preventDefault(); void apply(draft); } }}
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
              <>
              <SettingRow
                title={tr('OpenAI 端口', 'OpenAI port')}
                detail={listeningPort == null ? tr('OpenAI 兼容 HTTP 服务端口。', 'OpenAI-compatible HTTP server port.')
                  : `http://127.0.0.1:${listeningPort}/v1`}
                trailing={
                  <input
                    aria-label={tr('OpenAI 端口', 'OpenAI port')}
                    className="server-number-input"
                    disabled={busy || !configurable}
                    max={65535}
                    min={1}
                    onBlur={() => void apply(draft)}
                    onKeyDown={(event) => { if (event.key === 'Enter') { event.preventDefault(); void apply(draft); } }}
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
              <SettingRow
                title={tr('Anthropic 端口', 'Anthropic port')}
                detail={anthropicPort == null
                  ? tr('当前服务尚未启用 Anthropic API。', 'The current service has not enabled the Anthropic API.')
                  : `http://127.0.0.1:${anthropicPort}/v1/messages`}
                trailing={<input
                  aria-label={tr('Anthropic 端口', 'Anthropic port')}
                  className="server-number-input"
                  disabled={busy || !configurable || anthropicPort == null}
                  max={65535} min={1} type="number" value={anthropicDraft}
                  onChange={(event) => setAnthropicDraft(event.target.value)}
                  onBlur={() => void applyAnthropic()}
                  onKeyDown={(event) => { if (event.key === 'Enter') { event.preventDefault(); void applyAnthropic(); } }}
                />}
              />
              </>
            )}
          </div>
          {applyState && <p className="server-apply-state" role={applyState.error ? 'alert' : 'status'}>{applyState.text}</p>}
        </TMPanel>
        <MemorySettingsPanel />
        <ContextManagementPanel />
        <PrefixCacheSettingsPanel />
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
      </div>
      <InferencePolicyPanel />
      <RuntimeProfilesPanel />
      <RemoteRoutingPanel />
    </section>
  );
}
