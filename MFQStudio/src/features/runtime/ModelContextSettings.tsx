import { useEffect, useState } from 'react';
import { useRuntime } from '../../app/RuntimeProvider';
import { useConnectionScope } from '../../app/useConnectionScope';
import { SettingRow } from '../../app/display';
import { errorMessage, formatNumber } from '../../app/formatters';
import { useSettings } from '../settings/SettingsProvider';
import { toast } from '../../stores/toastStore';
import { runtimeApi } from '../../shared/api/resources/runtime';

export function ModelContextSettings() {
  const { instances, runtime, reloadingInstances, reloadModelContext } = useRuntime();
  const connectionScope = useConnectionScope();
  const { tr, contextSize, setContextSize } = useSettings();
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
    const current = connectionScope();
    try {
      const result = await reloadModelContext(instanceId, size);
      if (!current()) return;
      const actual = result.max_context ?? size;
      setDrafts((current) => ({ ...current, [instanceId]: String(actual) }));
      toast.success(tr(`${model} 已重载，上下文 ${formatNumber(actual)} tokens`,
        `${model} reloaded with a ${formatNumber(actual)} token context`));
    } catch (cause) {
      if (current()) toast.error(errorMessage(cause));
    }
  }

  if (!loaded.length) return <SettingRow
    title={tr('最大上下文', 'Maximum context')}
    detail={tr('下一次加载模型的默认上下文。', 'Default context for the next model load.')}
    trailing={<div className="server-input-unit">
      <input aria-label={tr('默认最大上下文', 'Default maximum context')}
        className="server-number-input" min={512} max={1048576} type="number"
        value={contextSize} onChange={(event) => setContextSize(Number(event.target.value))} />
      <span>tokens</span>
    </div>} />;

  return <div className="model-context-settings">
    <div className="model-context-heading">{tr('最大上下文', 'Maximum context')}</div>
    {loaded.map((item) => {
      const capacity = item.context_capacity ?? capacities[item.id] ?? (runtime?.instance_id === item.id ? runtime.context_capacity : undefined);
      const value = drafts[item.id] ?? String(item.context_size ?? contextSize);
      const size = Number(value);
      const valid = Number.isSafeInteger(size) && size >= 512 && size <= (capacity ?? 1048576);
      const busy = reloadingInstances?.[item.id] != null;
      return <SettingRow key={item.id} title={item.model}
        detail={tr(`当前 ${formatNumber(item.context_size ?? contextSize)} tokens${capacity ? ` · 上限 ${formatNumber(capacity)}` : ''}`,
          `Current ${formatNumber(item.context_size ?? contextSize)} tokens${capacity ? ` · Limit ${formatNumber(capacity)}` : ''}`)}
        trailing={<div className="server-row-actions">
          <div className="server-input-unit">
            <input aria-label={tr(`${item.model} 最大上下文`, `${item.model} maximum context`)}
              className="server-number-input" min={512} max={capacity ?? 1048576} type="number"
              value={value} disabled={busy}
              onChange={(event) => setDrafts((current) => ({ ...current, [item.id]: event.target.value }))} />
            <span>tokens</span>
          </div>
          <button aria-label={tr(`重载 ${item.model}`, `Reload ${item.model}`)}
            disabled={busy || !valid} onClick={() => void reload(item.id, size, item.model)} type="button">
            {busy ? tr('重载中…', 'Reloading…') : tr('重载', 'Reload')}
          </button>
        </div>} />;
    })}
  </div>;
}
