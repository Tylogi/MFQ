/** Configure model context sizes using reported capacities and server-side validation. */
import { localized } from '../../i18n/messages';
import { useTranslation } from 'react-i18next';
import { useEffect, useState } from 'react';
import { useRuntime } from '../../app/RuntimeProvider';
import { SettingRow } from '../../app/display';
import { errorMessage, formatNumber } from '../../app/formatters';
import { useSettings } from '../settings/SettingsProvider';
import { toast } from '../../stores/toastStore';
import { runtimeApi } from '../../shared/api/resources/runtime';

/** Edit default and loaded-model context budgets; reload explicitly after validation. */
export function ModelContextSettings() {
  const { instances, runtime, reloadingInstances, reloadModelContext } = useRuntime();
  const { contextSize, setContextSize } = useSettings();
  const { t } = useTranslation();
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const [capacities, setCapacities] = useState<Record<string, number>>({});
  const loaded = instances.filter((item) => item.state === 'ready' || item.state === 'busy');
  const missingCapacities = loaded.filter((item) => item.context_capacity == null).map((item) => item.id).sort().join('|');
  useEffect(() => {
    let disposed = false;
    const ids = missingCapacities ? missingCapacities.split('|') : [];
    void Promise.allSettled(ids.map(async (id) => {
      const status = await runtimeApi.runtimeStatus(id);
      if (!disposed && status.context_capacity != null) setCapacities((current) => ({ ...current, [id]: status.context_capacity! }));
    }));
    return () => { disposed = true; };
  }, [missingCapacities]);

  async function reload(instanceId: string, size: number, model: string) {
    try {
      const result = await reloadModelContext(instanceId, size);
      const actual = result.max_context ?? size;
      setDrafts((current) => ({ ...current, [instanceId]: String(actual) }));
      toast.success(localized('runtime:modelContextSettings.reloadedWithATokenContext', { model: model, tokens: formatNumber(actual) }));
    } catch (cause) {
      toast.error(errorMessage(cause));
    }
  }

  if (!loaded.length) return <SettingRow
    title={t('runtime:modelContextSettings.maximumContext')}
    detail={t('runtime:modelContextSettings.defaultContextForTheNextModelLoad')}
    trailing={<div className="server-input-unit">
      <input aria-label={t('runtime:modelContextSettings.defaultMaximumContext')}
        className="server-number-input" min={512} step={1} type="number"
        value={contextSize} onChange={(event) => setContextSize(Number(event.target.value))} />
      <span>tokens</span>
    </div>} />;

  return <div className="model-context-settings">
    <div className="model-context-heading">{t('runtime:modelContextSettings.maximumContext')}</div>
    {loaded.map((item) => {
      const capacity = item.context_capacity ?? capacities[item.id] ?? (runtime?.instance_id === item.id ? runtime.context_capacity : undefined);
      const value = drafts[item.id] ?? String(item.context_size ?? contextSize);
      const size = Number(value);
      const valid = Number.isSafeInteger(size) && size >= 512 && (capacity == null || size <= capacity);
      const busy = reloadingInstances?.[item.id] != null;
      return <SettingRow key={item.id} title={item.model}
        detail={capacity ? t('runtime:modelContextSettings.currentTokensLimit', { tokens: formatNumber(item.context_size ?? contextSize), limit: formatNumber(capacity) }) : t('runtime:modelContextSettings.currentTokens', { tokens: formatNumber(item.context_size ?? contextSize) })}
        trailing={<div className="server-row-actions">
          <div className="server-input-unit">
            <input aria-label={t('runtime:modelContextSettings.maximumContext2', { model: item.model })}
              className="server-number-input" min={512} max={capacity} step={1} type="number"
              value={value} disabled={busy}
              onChange={(event) => setDrafts((current) => ({ ...current, [item.id]: event.target.value }))} />
            <span>tokens</span>
          </div>
          <button aria-label={t('runtime:modelContextSettings.reload', { model: item.model })}
            disabled={busy || !valid} onClick={() => void reload(item.id, size, item.model)} type="button">
            {busy ? t('runtime:modelContextSettings.reloading') : t('common:reload')}
          </button>
        </div>} />;
    })}
  </div>;
}
