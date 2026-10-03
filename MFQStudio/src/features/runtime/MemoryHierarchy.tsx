/** Model colors are shared by every memory/storage tier. Missing telemetry is not zero. */
import { useRef } from 'react';
import type { RuntimeInstance } from '../../shared/api/types';
import { formatBytes, formatNumber } from '../../app/formatters';
import { Icon, TMPanel } from '../../app/display';
import { useSettings } from '../settings/SettingsProvider';

const MODEL_COLORS = ['var(--accent)', '#e07070', '#65ad83', '#d5ae58', '#a28bd0', '#5eafb5', '#d58dad'];
type Tier = 'weights' | 'kv' | 'experts' | 'ple';
const modelCount = (count: number) => `${count} ${count === 1 ? 'model' : 'models'}`;
const resourceBytes = (bytes: number) => bytes === 0 ? '0 B' :
  formatBytes(bytes).replace(/\b(KB|MB|GB|TB)\b/g, (unit) => `${unit[0]}iB`);

export function MemoryHierarchy({ instances, connectionRevision = 0 }: {
  instances: RuntimeInstance[];
  connectionRevision?: number;
}) {
  const { tr } = useSettings();
  const palette = useRef({ revision: connectionRevision, ids: [] as string[] });
  if (palette.current.revision !== connectionRevision) {
    palette.current = { revision: connectionRevision, ids: [] };
  }
  const loaded = instances.filter((item) => item.state === 'ready' || item.state === 'busy');
  // Register in load order, never in byte-size or selected-model order.
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
  const tiers: { id: Tier; title: string; detail: string }[] = [
    { id: 'weights', title: tr('常驻专家与稠密权重', 'Resident experts and dense weights'),
      detail: tr(`${loaded.length} 个模型`, modelCount(loaded.length)) },
    { id: 'kv', title: tr('常驻 KV Cache 与前缀缓存', 'Resident KV and prefix cache'),
      detail: loaded.every((item) => item.memory?.context_count != null && item.memory?.prefix_cache_blocks != null)
        ? tr(`${formatNumber(loaded.reduce((sum, item) => sum + item.memory!.context_count!, 0))} 组上下文，${formatNumber(loaded.reduce((sum, item) => sum + item.memory!.prefix_cache_blocks!, 0))} 个缓存块`,
          `${formatNumber(loaded.reduce((sum, item) => sum + item.memory!.context_count!, 0))} contexts · ${formatNumber(loaded.reduce((sum, item) => sum + item.memory!.prefix_cache_blocks!, 0))} cache blocks`)
        : tr('上下文与缓存块统计未上报', 'Context and cache-block counts not reported') },
    { id: 'experts', title: tr('SSD 流式加载专家', 'SSD-streamed experts'), detail: streamedCount('ssd_experts') },
    { id: 'ple', title: tr('SSD 流式加载 PLE 表', 'SSD-streamed PLE tables'), detail: streamedCount('ssd_ple') },
  ];
  function streamedCount(field: 'ssd_experts' | 'ssd_ple'): string {
    const known = loaded.filter((item) => item.memory?.[field] === true).length;
    const unknown = loaded.filter((item) => item.memory?.[field] == null).length;
    return tr(`${known} 个模型${unknown ? `，${unknown} 个未上报` : ''}`,
      `${modelCount(known)}${unknown ? ` · ${unknown} not reported` : ''}`);
  }
  return (
    <TMPanel className="overview-memory-panel">
      <div className="overview-memory-heading">
        <div>
          <h2>{tr('运行资源', 'Runtime resources')}</h2>
          <p>{tr('前两项为常驻内存，后两项为 SSD 上的流式权重；色块表示各模型占比。',
            'The first two tiers reside in memory; the last two are streamed from SSD. Colors show each model’s share.')}</p>
        </div>
      </div>
      <div className="memory-model-legend" aria-label={tr('模型颜色图例', 'Model color legend')}>
        {loaded.map((item) => (
          <span key={item.id} title={item.model} data-model-id={item.id}>
            <i style={{ backgroundColor: color(item) }} />{item.model}
          </span>
        ))}
        {!loaded.length && <span>{tr('暂无已加载模型', 'No loaded models')}</span>}
      </div>
      <div className="overview-memory-tiers">
        {tiers.map((tier) => {
          const amounts = loaded.map((item) => ({ item, bytes: value(item, tier.id) }));
          const unknown = amounts.some(({ bytes }) => bytes == null);
          const total = amounts.reduce((sum, { bytes }) => sum + (bytes ?? 0), 0);
          return (
            <div className="memory-tier" key={tier.id} data-tier={tier.id}>
              <div className="memory-tier-heading">
                <div><strong>{tier.title}</strong><small>{tier.detail}</small></div>
                <span>{unknown ? tr('明细未上报', 'Breakdown not reported') : resourceBytes(total)}</span>
              </div>
              <div className="memory-tier-track" aria-label={tier.title}>
                {amounts.filter(({ bytes }) => bytes != null && bytes > 0).map(({ item, bytes }) => (
                  <span key={item.id} data-model-id={item.id}
                    style={{ backgroundColor: color(item), width: `${bytes! / total * 100}%` }}
                    title={`${item.model} · ${resourceBytes(bytes!)}${unknown ? '' : ` · ${formatNumber(bytes! / total * 100, 1)}%`}`} />
                ))}
              </div>
              {unknown && <small className="memory-tier-notice"><Icon name="info" size={12} />
                {tr('缺失统计不计入色条；需要支持明细的后端。', 'Missing telemetry is excluded from the bar; a backend with breakdown support is required.')}
              </small>}
            </div>
          );
        })}
      </div>
    </TMPanel>
  );
}
