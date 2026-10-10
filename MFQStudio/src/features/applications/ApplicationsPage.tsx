import { useEffect, useState } from 'react';
import { Icon, ScreenHeader } from '../../app/display';
import { errorMessage } from '../../app/formatters';
import { useRuntime } from '../../app/RuntimeProvider';
import { getApiBaseUrl } from '../../shared/api/client';
import { runtimeApi } from '../../shared/api/resources/runtime';
import { openStudioExternal } from '../../shared/platform/studio';
import { CompactSelect } from '../../shared/ui/CompactSelect';
import { useSettings } from '../settings/SettingsProvider';
import { anthropicEndpoint, openAIEndpoint } from '../runtime/endpoint';
import { APPLICATIONS, applicationConfiguration } from './configuration';

export function ApplicationsPage() {
  const { tr } = useSettings();
  const { instances, selectedModel, connectionRevision, studio } = useRuntime();
  const [selected, setSelected] = useState('');
  const [aliases, setAliases] = useState<Record<string, string>>({});
  const [port, setPort] = useState<number | null>(null);
  const [copied, setCopied] = useState('');
  const [error, setError] = useState('');
  const available = instances.filter(item => item.state === 'ready' || item.state === 'busy').map(item => aliases[item.model] || item.model);
  const preferred = selected || aliases[selectedModel] || selectedModel;
  const model = available.includes(preferred) ? preferred : available[0] || '';
  const openAI = openAIEndpoint(studio?.service_url || getApiBaseUrl());
  const anthropic = anthropicEndpoint(studio?.service_url || getApiBaseUrl(), port);

  useEffect(() => {
    let disposed = false;
    setSelected(''); setAliases({}); setPort(null); setCopied(''); setError('');
    const refresh = async () => {
      const [listener, mapping] = await Promise.allSettled([runtimeApi.runtimeListener(), runtimeApi.modelAliases()]);
      if (disposed) return;
      if (listener.status === 'fulfilled') setPort(listener.value.anthropic_port ?? null);
      if (mapping.status === 'fulfilled') setAliases(mapping.value.aliases);
    };
    void refresh();
    window.addEventListener('focus', refresh);
    return () => { disposed = true; window.removeEventListener('focus', refresh); };
  }, [connectionRevision]);

  async function copy(id: string, value: string) {
    try { await navigator.clipboard.writeText(value); setCopied(id); setError(''); }
    catch (cause) { setError(errorMessage(cause)); }
  }

  return <section className="dashboard-view applications-page">
    <ScreenHeader title={tr('应用', 'Applications')} subtitle={tr('复制接入参数与配置片段，在应用中使用 MFQ。', 'Copy connection details and configuration snippets to use MFQ in applications.')} />
    {error && <div className="inline-error" role="alert">{error}</div>}
    <div className="application-parameters">
      <div className="application-model-row"><label htmlFor="application-model">{tr('模型', 'Model')}</label>
        <CompactSelect id="application-model" value={model} disabled={!available.length} onChange={event => { setSelected(event.target.value); setCopied(''); }}>
          {!available.length && <option value="">{tr('请先载入模型', 'Load a model first')}</option>}
          {available.map(name => <option key={name} value={name}>{name}</option>)}
        </CompactSelect>
        <button type="button" disabled={!model} onClick={() => void copy('model', model)}><Icon name={copied === 'model' ? 'check' : 'copy'} />{tr('复制模型 ID', 'Copy model ID')}</button>
      </div>
      {[['openai', 'OpenAI', openAI], ['anthropic', 'Anthropic', anthropic]].map(([id, label, url]) => <div className="application-endpoint-row" key={id}>
        <span>{label}</span><code title={url || ''}>{url || tr('未启用', 'Disabled')}</code>
        <button type="button" aria-label={tr(`复制 ${label} 地址`, `Copy ${label} URL`)} disabled={!url} onClick={() => void copy(id!, url!)}><Icon name={copied === id ? 'check' : 'copy'} /></button>
      </div>)}
    </div>
    <p className="application-note">{tr('配置片段需手动合并，保留应用现有设置。YOUR_MFQ_API_KEY 替换为服务密钥；未启用鉴权时可填 mfq。', 'Merge snippets manually, retaining existing settings. Replace YOUR_MFQ_API_KEY with your service key; use mfq when authentication is disabled.')}</p>
    <div className="application-snippets">
      {APPLICATIONS.map(app => {
        const snippet = applicationConfiguration(app.id, model, openAI, anthropic);
        return <section className="application-entry" key={app.id}>
          <div className="application-entry-heading"><span className="application-mark" aria-hidden="true">{app.name[0]}</span>
            <div><h2>{app.name}</h2><small>{app.protocol}</small></div>
            <button type="button" onClick={() => void openStudioExternal(app.url).catch(cause => setError(errorMessage(cause)))}>{tr('官网', 'Website')}<Icon name="link" size={14} /></button>
            <button type="button" disabled={!model || !snippet} onClick={() => void copy(app.id, snippet)}><Icon name={copied === app.id ? 'check' : 'copy'} />{copied === app.id ? tr('已复制', 'Copied') : tr('复制配置', 'Copy configuration')}</button>
          </div>
          <details><summary>{tr('配置片段', 'Configuration snippet')}<code>{app.file}</code></summary>
            {app.id === 'dsh' && <p>{tr('桌面版使用 desktop 而非 web 目录；在应用环境或凭据中设置 MFQ_API_KEY。JSON 数组可作为 YAML 配置片段。', 'Use the desktop directory instead of web for the desktop app. Set MFQ_API_KEY in the application environment or credentials. The JSON array is also a valid YAML snippet.')}</p>}
            {model && snippet ? <pre>{snippet}</pre> : <p>{tr('请先载入模型，并启用对应 API 端口。', 'Load a model and enable the required API listener first.')}</p>}
          </details>
        </section>;
      })}
    </div>
  </section>;
}
