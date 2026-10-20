import { useEffect, useRef, useState } from 'react';
import { SettingRow } from '../../app/display';
import { errorMessage } from '../../app/formatters';
import { useRuntime } from '../../app/RuntimeProvider';
import { getApiBaseUrl, getApiToken, setApiToken } from '../../shared/api/client';
import { runtimeApi, type RuntimeAuthenticationPolicy } from '../../shared/api/resources/runtime';
import { saveStudioCredential } from '../../studio';
import { useSettings } from '../settings/SettingsProvider';

export function ApiAuthenticationPanel() {
  const { tr } = useSettings();
  const { connectionRevision } = useRuntime();
  const [policy, setPolicy] = useState<RuntimeAuthenticationPolicy | null>(null);
  const [enabled, setEnabled] = useState(false);
  const [key, setKey] = useState('');
  const [saving, setSaving] = useState(false);
  const [status, setStatus] = useState<{ error: boolean; text: string } | null>(null);
  const mounted = useRef(false);
  const pending = useRef(false);
  useEffect(() => {
    mounted.current = true;
    let disposed = false;
    setPolicy(null); setEnabled(false); setKey(''); setStatus(null);
    void runtimeApi.authenticationPolicy().then(value => {
      if (!disposed) { setPolicy(value); setEnabled(value.enabled); }
    }).catch(cause => {
      if (!disposed) setStatus({ error: true, text: errorMessage(cause, tr) });
    });
    return () => { disposed = true; mounted.current = false; };
  }, [connectionRevision]);

  async function apply() {
    if (pending.current || !policy || (enabled === policy.enabled && !key)) return;
    if (enabled && !key && !policy.enabled) {
      setEnabled(policy.enabled); setKey('');
      setStatus({ error: true, text: tr('请输入 API Key 后启用鉴权。', 'Enter an API key to enable authentication.') });
      return;
    }
    if (enabled && key && (key.length > 512 || !/^[\x21-\x7e]+$/.test(key))) {
      setEnabled(policy.enabled); setKey('');
      setStatus({ error: true, text: tr('API Key 必须为 1–512 个可打印英文字符，不能包含空格。', 'API key must contain 1–512 printable ASCII characters without spaces.') });
      return;
    }
    const endpoint = getApiBaseUrl();
    const previous = getApiToken();
    const current = () => endpoint === getApiBaseUrl() && previous === getApiToken();
    const next = enabled ? key || previous : '';
    const credentialChanged = next !== previous;
    let credentialSaved = false;
    pending.current = true; setSaving(true); setStatus(null);
    try {
      if (credentialChanged) {
        await saveStudioCredential(next);
        credentialSaved = true;
      }
      if (!current()) return;
      const accepted = await runtimeApi.configureAuthenticationPolicy(enabled, enabled && key ? key : undefined);
      if (!current()) return;
      setApiToken(next);
      if (mounted.current) {
        setPolicy(accepted); setEnabled(accepted.enabled); setKey('');
        setStatus({ error: false, text: tr('鉴权已应用，无需重载模型。', 'Authentication applied without reloading models.') });
      }
    } catch (cause) {
      if (credentialSaved && current()) {
        try { await saveStudioCredential(previous); } catch {}
      }
      if (mounted.current && current()) {
        setEnabled(policy.enabled); setKey('');
        setStatus({ error: true, text: errorMessage(cause, tr) });
      }
    } finally {
      pending.current = false;
      if (mounted.current) setSaving(false);
    }
  }

  return <>
    <SettingRow title={tr('鉴权', 'Authentication')}
      detail={tr('OpenAI 与 Anthropic API 共用；应用后立即生效，无需重载模型。', 'Shared by the OpenAI and Anthropic APIs; applies immediately without reloading models.')}
      trailing={<div className="server-row-actions">
        <select aria-label={tr('鉴权', 'Authentication')} disabled={!policy || saving} value={enabled ? 'api-key' : 'none'}
          onChange={event => { setEnabled(event.target.value === 'api-key'); setKey(''); setStatus(null); }}>
          <option value="none">{tr('无', 'None')}</option><option value="api-key">API Key</option>
        </select>
        <button type="button" disabled={!policy || saving || (enabled === policy.enabled && !key)} onClick={() => void apply()}>
          {saving ? tr('应用中…', 'Applying…') : tr('应用鉴权', 'Apply authentication')}
        </button>
      </div>} />
    {enabled && <SettingRow title="API Key"
      detail={tr('服务端仅保存密钥哈希；应用端通过系统凭据库保存，浏览器端仅保留在内存。', 'The server stores only the key hash; Studio uses the system credential vault, while the browser keeps the key in memory.')}
      trailing={<input aria-label="API Key" className="server-wide-input" type="password" autoComplete="new-password" spellCheck={false}
        disabled={saving} value={key} maxLength={512}
        placeholder={policy?.enabled && policy.has_key ? tr('已设置，留空保持不变', 'Configured; leave blank to keep') : tr('输入 API Key', 'Enter API key')}
        onChange={event => setKey(event.target.value)}
        onKeyDown={event => { if (event.key === 'Enter') { event.preventDefault(); void apply(); } }} />} />}
    {status && <p className="server-apply-state" role={status.error ? 'alert' : 'status'}>{status.text}</p>}
  </>;
}
