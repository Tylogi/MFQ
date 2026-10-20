import { useEffect, useRef, useState, type FormEvent } from 'react';
import { EyeIcon, EyeSlashIcon } from '@phosphor-icons/react';
import { useSettings } from '../features/settings/SettingsProvider';
import { ApiError, getApiBaseUrl, getApiToken, setApiToken, verifyApiToken } from '../shared/api/client';
import { saveStudioCredential } from '../studio';
import { useRuntime } from './RuntimeProvider';
import mfqMark from '../assets/mfq-mark.svg';

export function LoginPage() {
  const { tr } = useSettings();
  const { reloadService } = useRuntime();
  const [key, setKey] = useState('');
  const [visible, setVisible] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const pending = useRef(false);
  const controller = useRef<AbortController | null>(null);
  const endpoint = getApiBaseUrl();

  useEffect(() => () => { controller.current?.abort(); }, []);

  async function signIn(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (pending.current) return;
    const token = key.trim();
    if (!token || token.length > 512 || !/^[\x21-\x7e]+$/.test(token)) {
      setError(tr('请输入有效的 API Key，不能包含空格。', 'Enter a valid API key without spaces.'));
      return;
    }
    const previous = getApiToken();
    const request = new AbortController();
    controller.current = request;
    const current = () => !request.signal.aborted && endpoint === getApiBaseUrl() && previous === getApiToken();
    pending.current = true;
    setSubmitting(true);
    setError(null);
    let verified = false;
    try {
      await verifyApiToken(token, request.signal);
      if (!current()) return;
      verified = true;
      await saveStudioCredential(token);
      if (!current()) return;
      setApiToken(token);
      if (!await reloadService() && !request.signal.aborted) {
        setError(tr('密钥已验证，但工作区未能加载。请检查连接及密钥权限后重试。', 'The key was verified, but the workspace could not load. Check the connection and key permissions, then retry.'));
      }
    } catch (cause) {
      if (!current()) return;
      setError(cause instanceof ApiError && cause.status === 401
        ? tr('API Key 不正确或已失效，请重新输入。', 'The API key is incorrect or expired. Please try again.')
        : cause instanceof ApiError && cause.status === 403
          ? tr('此密钥没有访问权限，请使用有相应权限的 API Key。', 'This key does not have access. Use an API key with the required permissions.')
          : verified
            ? tr('无法保存登录凭据，请检查系统凭据库后重试。', 'Could not save the sign-in credential. Check the system credential vault and try again.')
            : tr('暂时无法连接服务，请检查网络后重试。', 'Could not reach the service. Check your connection and try again.'));
    } finally {
      pending.current = false;
      if (!request.signal.aborted) setSubmitting(false);
    }
  }

  return (
    <main className="login-page">
      <section className="login-card" aria-labelledby="login-title">
        <div className="login-brand"><img src={mfqMark} alt="" /><span>MFQ Studio</span></div>
        <h1 id="login-title">{tr('登录', 'Sign in')}</h1>
        <p className="login-description">{tr('此服务已启用鉴权，请输入 API Key。', 'This service requires authentication. Enter your API key to continue.')}</p>
        <p className="login-endpoint" title={endpoint || window.location.origin}>{endpoint || window.location.origin}</p>
        <form onSubmit={event => void signIn(event)} aria-busy={submitting}>
          <label htmlFor="login-api-key">API Key</label>
          <div className="login-key-field">
            <input id="login-api-key" name="api-key" type={visible ? 'text' : 'password'} value={key}
              autoComplete="current-password" autoFocus spellCheck={false} autoCapitalize="none" maxLength={512}
              disabled={submitting} aria-invalid={Boolean(error)} aria-describedby={error ? 'login-error' : undefined}
              placeholder={tr('输入 API Key', 'Enter API key')}
              onChange={event => { setKey(event.target.value); setError(null); }} />
            <button type="button" className="login-key-visibility" aria-label={visible ? tr('隐藏密钥', 'Hide key') : tr('显示密钥', 'Show key')}
              aria-pressed={visible} onClick={() => setVisible(value => !value)}>
              {visible ? <EyeSlashIcon size={18} /> : <EyeIcon size={18} />}
            </button>
          </div>
          {error && <p id="login-error" className="login-error" role="alert">{error}</p>}
          <button className="login-submit" type="submit" disabled={submitting || !key.trim()}>
            {submitting ? tr('正在登录…', 'Signing in…') : tr('登录', 'Sign in')}
          </button>
        </form>
      </section>
    </main>
  );
}
