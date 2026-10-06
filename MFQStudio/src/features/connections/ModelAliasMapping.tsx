/** Provide ModelAliasMapping interface behavior. */
import { localized } from '../../i18n/messages';
import { useTranslation } from 'react-i18next';
import { useEffect, useRef, useState } from 'react';
import { Dialog } from '../../shared/ui/Dialog';
import { runtimeApi } from '../../shared/api/resources/runtime';
import { errorMessage } from '../../app/formatters';
import { toast } from '../../stores/toastStore';

/** Edit model aliases exposed by the server and persist the validated mapping. */
export function ModelAliasMapping({ models, selectedModel }: { models: string[]; selectedModel: string }) {
  const { t } = useTranslation();
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
      toast.success(localized('connections:modelAliasMapping.aliasMappingSaved'));
    } catch (cause) { toast.error(errorMessage(cause)); }
    finally { setSaving(false); }
  }
  const names = [...new Set([...models, ...Object.keys(aliases)])];
  return <>
    <strong title={selectedModel}>{aliases[selectedModel] || selectedModel || t('connections:modelAliasMapping.notLoaded')}</strong>
    <button ref={trigger} disabled={!ready} onClick={() => setOpen(true)} type="button">{t('connections:modelAliasMapping.aliasMapping')}</button>
    <Dialog open={open} onOpenChange={setOpen} title={t('connections:modelAliasMapping.aliasMapping')} closeLabel={t('common:close')}
      description={t('connections:modelAliasMapping.setModelIdsForOpenaiCompatibleEndpointsLeaveBlankToUseThe')}
      className="model-alias-dialog" returnFocusRef={trigger}>
      <div className="model-alias-list">{names.map((name) => <label key={name}>
        <span title={name}>{name}</span><input aria-label={`${name} ${t('connections:modelAliasMapping.alias')}`} placeholder={name}
          disabled={saving} maxLength={255} value={aliases[name] || ''} onChange={(event) => setAliases((current) => ({ ...current, [name]: event.target.value }))} />
      </label>)}</div>
      <footer><button disabled={saving || !names.length} onClick={() => void save()} type="button">{saving ? t('connections:modelAliasMapping.saving') : t('common:save')}</button></footer>
    </Dialog>
  </>;
}
