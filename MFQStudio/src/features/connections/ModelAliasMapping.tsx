import { useEffect, useRef, useState } from 'react';
import { Dialog } from '../../shared/ui/Dialog';
import { runtimeApi } from '../../shared/api/resources/runtime';
import { useSettings } from '../settings/SettingsProvider';
import { errorMessage } from '../../app/formatters';
import { toast } from '../../stores/toastStore';

export function ModelAliasMapping({ models, selectedModel }: { models: string[]; selectedModel: string }) {
  const { tr } = useSettings();
  const [open, setOpen] = useState(false);
  const [aliases, setAliases] = useState<Record<string, string>>({});
  const [ready, setReady] = useState(false);
  const [saving, setSaving] = useState(false);
  const trigger = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    let disposed = false;
    void runtimeApi.modelAliases().then((result) => {
      if (!disposed) { setAliases(result.aliases); setReady(true); }
    }).catch(() => {});
    return () => { disposed = true; };
  }, []);
  async function save() {
    setSaving(true);
    try {
      const result = await runtimeApi.configureModelAliases(aliases);
      setAliases(result.aliases);
      setOpen(false);
      toast.success(tr('别名映射已保存', 'Alias mapping saved'));
    } catch (cause) { toast.error(errorMessage(cause)); }
    finally { setSaving(false); }
  }
  const names = [...new Set([...models, ...Object.keys(aliases)])];
  return <>
    <strong title={selectedModel}>{aliases[selectedModel] || selectedModel || tr('尚未加载', 'Not loaded')}</strong>
    <button ref={trigger} disabled={!ready} onClick={() => setOpen(true)} type="button">{tr('别名映射', 'Alias mapping')}</button>
    <Dialog open={open} onOpenChange={setOpen} title={tr('别名映射', 'Alias mapping')} closeLabel={tr('关闭', 'Close')}
      description={tr('为 OpenAI 兼容端点设置模型 ID，留空使用原模型名。', 'Set model IDs for OpenAI-compatible endpoints. Leave blank to use the original name.')}
      className="model-alias-dialog" returnFocusRef={trigger}>
      <div className="model-alias-list">{names.map((name) => <label key={name}>
        <span title={name}>{name}</span><input aria-label={`${name} ${tr('别名', 'alias')}`} placeholder={name}
          disabled={saving} maxLength={255} value={aliases[name] || ''} onChange={(event) => setAliases((current) => ({ ...current, [name]: event.target.value }))} />
      </label>)}</div>
      <footer><button disabled={saving || !names.length} onClick={() => void save()} type="button">{saving ? tr('保存中…', 'Saving…') : tr('保存', 'Save')}</button></footer>
    </Dialog>
  </>;
}
