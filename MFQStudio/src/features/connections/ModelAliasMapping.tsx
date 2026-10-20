import { useEffect, useRef, useState } from 'react';
import { Dialog } from '../../shared/ui/Dialog';
import { useConnectionScope } from '../../app/useConnectionScope';
import { runtimeApi } from '../../shared/api/resources/runtime';
import { useSettings } from '../settings/SettingsProvider';
import { errorMessage } from '../../app/formatters';
import { toast } from '../../stores/toastStore';

export function ModelAliasMapping({ models, selectedModel }: { models: string[]; selectedModel: string }) {
  const { tr } = useSettings();
  const connectionScope = useConnectionScope();
  const [open, setOpen] = useState(false);
  const [aliases, setAliases] = useState<Record<string, string>>({});
  const [ready, setReady] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const applied = useRef<Record<string, string>>({});
  const trigger = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    let disposed = false;
    void runtimeApi.modelAliases().then((result) => {
      if (!disposed) { applied.current = result.aliases; setAliases(result.aliases); setReady(true); }
    }).catch(() => {});
    return () => { disposed = true; };
  }, []);
  async function save() {
    const normalized = Object.entries(aliases).filter(([name, alias]) => alias.trim() && alias.trim() !== name).map(([name, alias]) => [name, alias.trim()]);
    const invalid = normalized.some(([, alias]) => alias.length > 255 || /[\x00-\x1f]/.test(alias));
    const duplicate = new Set(normalized.map(([, alias]) => alias)).size !== normalized.length || normalized.some(([, alias]) => models.includes(alias));
    if (invalid || duplicate) {
      setAliases(applied.current);
      setError(invalid ? tr('模型别名不能包含控制字符，长度不能超过 255 个字符。', 'Model aliases cannot contain control characters or exceed 255 characters.') : tr('模型别名不能重复，也不能与其它模型的名称相同。', 'Model aliases must be unique and cannot match another model’s name.'));
      return;
    }
    const current = connectionScope();
    setSaving(true); setError('');
    try {
      const result = await runtimeApi.configureModelAliases(aliases);
      if (!current()) return;
      applied.current = result.aliases;
      setAliases(result.aliases);
      setOpen(false);
      toast.success(tr('别名映射已保存', 'Alias mapping saved'));
    } catch (cause) { if (current()) { setAliases(applied.current); setError(errorMessage(cause, tr)); } }
    finally { if (current()) setSaving(false); }
  }
  const names = [...new Set([...models, ...Object.keys(aliases)])];
  return <>
    <strong title={selectedModel}>{applied.current[selectedModel] || selectedModel || tr('尚未加载', 'Not loaded')}</strong>
    <button ref={trigger} disabled={!ready} onClick={() => setOpen(true)} type="button">{tr('别名映射', 'Alias mapping')}</button>
    <Dialog open={open} onOpenChange={setOpen} title={tr('别名映射', 'Alias mapping')} closeLabel={tr('关闭', 'Close')}
      description={tr('为 OpenAI 兼容端点设置模型 ID，留空使用原模型名。', 'Set model IDs for OpenAI-compatible endpoints. Leave blank to use the original name.')}
      className="model-alias-dialog" returnFocusRef={trigger}>
      <div className="model-alias-list">{names.map((name) => <label key={name}>
        <span title={name}>{name}</span><input aria-label={`${name} ${tr('别名', 'alias')}`} placeholder={name}
          disabled={saving} maxLength={255} value={aliases[name] || ''} onChange={(event) => setAliases((current) => ({ ...current, [name]: event.target.value }))} />
      </label>)}</div>
      {error && <p className="server-apply-state" role="alert">{error}</p>}
      <footer><button disabled={saving || !names.length} onClick={() => void save()} type="button">{saving ? tr('保存中…', 'Saving…') : tr('保存', 'Save')}</button></footer>
    </Dialog>
  </>;
}
