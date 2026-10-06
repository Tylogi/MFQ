/** Display per-model memory residency, cache usage, and streamed weight tiers. */
import { useTranslation } from 'react-i18next';
import { useRef } from 'react';
import type { RuntimeInstance } from '../../shared/api/types';
import { formatBytes, formatNumber } from '../../app/formatters';
import { Icon, TMPanel } from '../../app/display';
import { ModelVendorMark } from '../../app/ModelVendorMark';

const MODEL_COLORS = ['var(--accent)', '#e07070', '#65ad83', '#d5ae58', '#a28bd0', '#5eafb5', '#d58dad'];
type Tier = 'weights' | 'kv' | 'experts' | 'ple';
const resourceBytes = (bytes: number) => bytes === 0 ? '0 B' :
  formatBytes(bytes).replace(/\b(KB|MB|GB|TB)\b/g, (unit) => `${unit[0]}iB`);

/** Display per-model memory residency, cache usage, and streamed weight tiers. */
export function MemoryHierarchy({ instances, memoryCapacityBytes, connectionRevision = 0 }: {
  instances: RuntimeInstance[];
  memoryCapacityBytes?: number | null;
  connectionRevision?: number;
}) {
  const { t } = useTranslation();
  const palette = useRef({ revision: connectionRevision, ids: [] as string[] });
  if (palette.current.revision !== connectionRevision) {
    palette.current = { revision: connectionRevision, ids: [] };
  }
  const loaded = instances.filter((item) => item.state === 'ready' || item.state === 'busy');
  [...loaded].sort((a, b) => (a.started_at || '').localeCompare(b.started_at || '')).forEach((item) => {
    if (!palette.current.ids.includes(item.model)) palette.current.ids.push(item.model);
  });
  loaded.sort((a, b) => palette.current.ids.indexOf(a.model) - palette.current.ids.indexOf(b.model));
  const color = (item: RuntimeInstance) => {
    const index = palette.current.ids.indexOf(item.model);
    return MODEL_COLORS[index] ?? `hsl(${(index * 137.508) % 360} 45% 60%)`;
  };
  const value = (item: RuntimeInstance, tier: Tier): number | null => {
    const memory = item.memory;
    if (!memory) return null;
    if (tier === 'weights') return memory.resident_weight_bytes;
    if (tier === 'kv') return memory.kv_bytes;
    if (tier === 'experts') return memory.ssd_experts === false ? 0 : memory.ssd_expert_bytes;
    return memory.ssd_ple === false ? 0 : memory.ssd_ple_bytes;
  };
  const weightAmounts = loaded.map((item) => value(item, 'weights'));
  const weightTotal = weightAmounts.reduce<number>((sum, bytes) => sum + (bytes ?? 0), 0);
  const memoryCapacity = memoryCapacityBytes != null && Number.isFinite(memoryCapacityBytes) && memoryCapacityBytes >= 0
    ? memoryCapacityBytes : null;
  const cacheCapacity = memoryCapacity != null && weightAmounts.every((bytes) => bytes != null)
    ? Math.max(0, memoryCapacity - weightTotal) : null;
  const tiers: { id: Tier; title: string; detail: string }[] = [
    { id: 'weights', title: t('runtime:memoryHierarchy.residentExpertsAndDenseWeights'),
      detail: t('runtime:memoryHierarchy.models', { count: loaded.length }) },
    { id: 'kv', title: t('runtime:memoryHierarchy.residentKvAndPrefixCache'),
      detail: loaded.every((item) => item.memory?.context_count != null && item.memory?.prefix_cache_blocks != null)
        ? t('runtime:memoryHierarchy.contextsCacheBlocks', { contexts: formatNumber(loaded.reduce((sum, item) => sum + item.memory!.context_count!, 0)), blocks: formatNumber(loaded.reduce((sum, item) => sum + item.memory!.prefix_cache_blocks!, 0)) })
        : t('runtime:memoryHierarchy.contextAndCacheBlockCountsNotReported') },
    { id: 'experts', title: t('runtime:memoryHierarchy.ssdStreamedExperts'), detail: streamedCount('ssd_experts') },
    { id: 'ple', title: t('runtime:memoryHierarchy.ssdStreamedPleTables'), detail: streamedCount('ssd_ple') },
  ];
  function streamedCount(field: 'ssd_experts' | 'ssd_ple'): string {
    const known = loaded.filter((item) => item.memory?.[field] === true).length;
    const unknown = loaded.filter((item) => item.memory?.[field] == null).length;
    return unknown
      ? t('runtime:memoryHierarchy.modelsNotReported', { count: known, unknown: unknown })
      : t('runtime:memoryHierarchy.models2', { count: known });
  }
  return (
    <TMPanel className="overview-memory-panel">
      <div className="overview-memory-heading">
        <div>
          <h2>{t('runtime:memoryHierarchy.runtimeResources')}</h2>
          <p>{t('runtime:memoryHierarchy.theFirstTwoTiersResideInMemoryTheLastTwoAreStreamed')}</p>
        </div>
      </div>
      <div className="memory-model-legend" aria-label={t('runtime:memoryHierarchy.modelColorLegend')}>
        {loaded.map((item) => (
          <span key={item.id} title={item.memory?.wired_available == null ? item.model :
            item.memory.wired_available
              ? `${item.model} · Metal ${t('runtime:memoryHierarchy.wired')} ${item.memory.wired_bytes == null ? '--' : resourceBytes(item.memory.wired_bytes)} / ${item.memory.wired_limit_bytes == null ? '--' : resourceBytes(item.memory.wired_limit_bytes)}`
              : `${item.model} · ${t('runtime:memoryHierarchy.metalMemoryWiringUnavailable')}`} data-model-id={item.id}>
            <i style={{ backgroundColor: color(item) }} />{item.model}
            <ModelVendorMark name={item.model} size={18} />
          </span>
        ))}
        {!loaded.length && <span>{t('runtime:memoryHierarchy.noLoadedModels')}</span>}
      </div>
      <div className="overview-memory-tiers">
        {tiers.map((tier) => {
          const amounts = loaded.map((item) => ({ item, bytes: value(item, tier.id) }));
          const missing = amounts.filter(({ bytes }) => bytes == null);
          const unknown = missing.length > 0;
          const reported = amounts.length - missing.length;
          const total = amounts.reduce((sum, { bytes }) => sum + (bytes ?? 0), 0);
          const resident = tier.id === 'weights' || tier.id === 'kv';
          const capacity = tier.id === 'weights' ? memoryCapacity : tier.id === 'kv' ? cacheCapacity : total;
          const scale = capacity == null ? 0 : Math.max(capacity, total);
          const used = !reported && unknown ? t('runtime:memoryHierarchy.breakdownNotReported')
            : `${unknown ? '≥ ' : ''}${resourceBytes(total)}`;
          return (
            <div className="memory-tier" key={tier.id} data-tier={tier.id}>
              <div className="memory-tier-heading">
                <div><strong>{tier.title}</strong><small>{tier.detail}</small></div>
                <span className={unknown ? 'memory-tier-unavailable' : undefined}>
                  {resident ? `${used} / ${capacity == null ? '--' : resourceBytes(capacity)}` : used}
                </span>
              </div>
              <div className="memory-tier-track" aria-label={tier.title}>
                {amounts.filter(({ bytes }) => scale > 0 && bytes != null && bytes > 0).map(({ item, bytes }) => (
                  <span key={item.id} data-model-id={item.id}
                    style={{ backgroundColor: color(item), width: `${bytes! / scale * 100}%` }}
                    title={`${item.model} · ${resourceBytes(bytes!)}${unknown || !capacity ? '' : ` · ${formatNumber(bytes! / capacity * 100, 1)}%`}`} />
                ))}
              </div>
              {tier.id === 'kv' && <small className="memory-tier-prefix-quota">
                {t('runtime:memoryHierarchy.prefixCacheWithinThisTier')}{' '}
                {loaded.every((item) => item.memory?.prefix_cache_bytes != null)
                  ? resourceBytes(loaded.reduce((sum, item) => sum + item.memory!.prefix_cache_bytes!, 0)) : '--'}
                {' / '}{loaded.every((item) => item.memory?.prefix_cache_limit_bytes != null)
                  ? resourceBytes(loaded.reduce((sum, item) => sum + item.memory!.prefix_cache_limit_bytes!, 0)) : '--'}
              </small>}
              {tier.id === 'weights' && loaded.some((item) => item.memory?.wired_available === false) &&
                <small className="memory-tier-notice"><Icon name="info" size={12} />
                  {t('runtime:memoryHierarchy.metalMemoryWiringIsUnavailableForSomeModelsWeightsMayBePaged')}
                </small>}
              {unknown && <small className="memory-tier-notice"><Icon name="info" size={12} />
                {t('runtime:memoryHierarchy.breakdownNotReported2', { models: missing.map(({ item }) => item.model).join(', ') })}
              </small>}
            </div>
          );
        })}
      </div>
    </TMPanel>
  );
}
