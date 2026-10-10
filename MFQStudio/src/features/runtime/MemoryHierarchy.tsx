import { useRef, type ReactNode } from 'react';
import type { RuntimeInstance, RuntimeStatus } from '../../shared/api/types';
import { formatBytes, formatNumber } from '../../app/formatters';
import { Icon, TMPanel } from '../../app/display';
import { useSettings } from '../settings/SettingsProvider';
import { ModelVendorMark } from '../../app/ModelVendorMark';
import { hasSeparateVram } from './memoryArchitecture';

type Tier = 'weights' | 'kv' | 'ram-experts' | 'ram-kv' | 'experts' | 'ple' | 'ssd-kv';
const modelCount = (count: number) => `${count} ${count === 1 ? 'model' : 'models'}`;
const resourceBytes = (bytes: number) => bytes < 1024 ? `${formatNumber(bytes)} B` :
  formatBytes(bytes).replace(/\b(KB|MB|GB|TB)\b/g, (unit) => `${unit[0]}iB`);

export function MemoryHierarchy({ instances, memoryCapacityBytes, runtime, connectionRevision = 0, detailed = false, summary }: {
  instances: RuntimeInstance[];
  runtime?: RuntimeStatus | null;
  memoryCapacityBytes?: number | null;
  connectionRevision?: number;
  detailed?: boolean;
  summary?: ReactNode;
}) {
  const { tr } = useSettings();
  const separate = hasSeparateVram(runtime);
  const loadOrder = useRef({ revision: connectionRevision, ids: [] as string[] });
  if (loadOrder.current.revision !== connectionRevision) {
    loadOrder.current = { revision: connectionRevision, ids: [] };
  }
  const loaded = instances.filter((item) => item.state === 'ready' || item.state === 'busy' || item.state === 'unloading');
  [...loaded].sort((a, b) => (a.started_at || '').localeCompare(b.started_at || '')).forEach((item) => {
    if (!loadOrder.current.ids.includes(item.model)) loadOrder.current.ids.push(item.model);
  });
  loaded.sort((a, b) => loadOrder.current.ids.indexOf(a.model) - loadOrder.current.ids.indexOf(b.model));
  const value = (item: RuntimeInstance, tier: Tier): number | null => {
    const memory = item.memory;
    if (!memory) return null;
    if (tier === 'weights') return memory.resident_weight_bytes;
    if (tier === 'kv') return memory.kv_bytes;
    if (tier === 'ram-experts') return memory.ram_experts === false ? 0 : memory.ram_expert_bytes ?? null;
    if (tier === 'ram-kv') return memory.ram_kv === false ? 0 : memory.ram_kv_bytes ?? null;
    if (tier === 'experts') return memory.ssd_experts === false ? 0 : memory.ssd_expert_bytes;
    if (tier === 'ssd-kv') return memory.ssd_kv === false ? 0 : memory.ssd_kv_bytes ?? null;
    return memory.ssd_ple === false ? 0 : memory.ssd_ple_bytes;
  };
  const weightAmounts = loaded.map((item) => value(item, 'weights'));
  const weightTotal = weightAmounts.reduce<number>((sum, bytes) => sum + (bytes ?? 0), 0);
  const memoryCapacity = memoryCapacityBytes != null && Number.isFinite(memoryCapacityBytes) && memoryCapacityBytes >= 0
    ? memoryCapacityBytes : null;
  const cacheCapacity = memoryCapacity != null && weightAmounts.every((bytes) => bytes != null)
    ? Math.max(0, memoryCapacity - weightTotal) : null;
  const contextCount = loaded.every((item) => item.memory?.context_count != null)
    ? formatNumber(loaded.reduce((sum, item) => sum + item.memory!.context_count!, 0)) : '--';
  const cacheBlocks = loaded.every((item) => item.memory?.prefix_cache_blocks != null)
    ? formatNumber(loaded.reduce((sum, item) => sum + item.memory!.prefix_cache_blocks!, 0)) : '--';
  const tiers: { id: Tier; title: string; detail: string }[] = [
    { id: 'weights', title: separate ? tr('显存常驻专家与稠密权重', 'VRAM-resident experts and dense weights') : tr('常驻专家与稠密权重', 'Resident experts and dense weights'),
      detail: tr(`${loaded.length} 个模型`, modelCount(loaded.length)) },
    { id: 'kv', title: separate ? tr('显存常驻 KV Cache 与前缀缓存', 'VRAM-resident KV and prefix cache') : tr('常驻 KV Cache 与前缀缓存', 'Resident KV and prefix cache'),
      detail: tr(`${contextCount} 组上下文，${cacheBlocks} 个缓存块`, `${contextCount} contexts · ${cacheBlocks} cache blocks`) },
    ...(separate ? [
      { id: 'ram-experts' as const, title: tr('内存流式加载专家', 'RAM-streamed experts'), detail: streamedCount('ram_experts') },
      { id: 'ram-kv' as const, title: tr('内存流式 KV', 'RAM-streamed KV'), detail: streamedCount('ram_kv') },
    ] : []),
    { id: 'experts', title: tr('SSD 流式加载专家', 'SSD-streamed experts'), detail: separate && loaded.length && loaded.every(item => item.memory?.ssd_experts === false)
      ? tr('当前设备尚未支持', 'Not yet supported on this device') : streamedCount('ssd_experts') },
    { id: 'ple', title: tr('SSD 流式加载 PLE 表', 'SSD-streamed PLE tables'), detail: streamedCount('ssd_ple') },
    { id: 'ssd-kv', title: tr('SSD 流式 KV', 'SSD-streamed KV'), detail: streamedCount('ssd_kv') },
  ];
  function streamedCount(field: 'ram_experts' | 'ram_kv' | 'ssd_experts' | 'ssd_ple' | 'ssd_kv'): string {
    const known = loaded.filter((item) => item.memory?.[field] === true).length;
    const unknown = loaded.filter((item) => item.memory?.[field] == null).length;
    return tr(`${known} 个模型${unknown ? `，${unknown} 个未上报` : ''}`,
      `${modelCount(known)}${unknown ? ` · ${unknown} not reported` : ''}`);
  }
  function streamingStats(items: RuntimeInstance[]) {
    const total = (field: 'ssd_kv_read_bytes' | 'ssd_kv_written_bytes' | 'ssd_kv_reads' | 'ssd_kv_hits') =>
      items.every((item) => item.memory?.[field] != null)
        ? items.reduce((sum, item) => sum + item.memory![field]!, 0) : null;
    const reads = total('ssd_kv_reads'), hits = total('ssd_kv_hits');
    const hitRate = reads != null && hits != null && reads + hits > 0
      ? `${formatNumber(hits / (reads + hits) * 100, 1)}%` : '--';
    return <div className="memory-tier-streaming-summary">
      {[{ label: tr('累计读取', 'Total read'), value: total('ssd_kv_read_bytes') },
        { label: tr('累计写入', 'Total written'), value: total('ssd_kv_written_bytes') }].map((metric) =>
        <span key={metric.label}>{metric.label} <b>{metric.value == null ? '--' : resourceBytes(metric.value)}</b></span>)}
      <span title={tr('按流式微块的 RAM 命中与 SSD 读取次数计算。', 'Calculated from streamed microblock RAM hits and SSD reads.')}>
        {tr('RAM 命中率', 'RAM hit rate')} <b>{hitRate}</b></span>
    </div>;
  }
  return (
    <TMPanel className={`overview-memory-panel${detailed ? ' resource-allocation-panel' : ''}`}>
      <div className="overview-memory-heading">
        <div>
          <h2>{detailed ? tr('资源分配', 'Resource allocation') : tr('运行资源', 'Runtime resources')}</h2>
          <p>{separate ? tr('显存承载常驻权重与热缓存；RAM 承载流式专家和 KV，KV 经 LRU 淘汰后溢出到 SSD。前缀缓存使用显存热层与 SSD 冷层。',
            'VRAM holds resident weights and hot caches. RAM holds streamed experts and KV; LRU-evicted KV spills to SSD. Prefix caching uses VRAM and SSD.') : tr('前两项为常驻内存，其余为 SSD 上的流式权重和 KV。',
            'The first two tiers reside in memory; the others are streamed weights and KV on SSD.')}</p>
        </div>
      </div>
      {summary}
      <div className="memory-model-legend" aria-label={tr('模型颜色图例', 'Model color legend')}>
        {loaded.map((item) => (
          <span key={item.id} title={item.memory?.wired_available == null ? item.model :
            item.memory.wired_available
              ? `${item.model} · Metal ${tr('锁页', 'wired')} ${item.memory.wired_bytes == null ? '--' : resourceBytes(item.memory.wired_bytes)} / ${item.memory.wired_limit_bytes == null ? '--' : resourceBytes(item.memory.wired_limit_bytes)}`
              : `${item.model} · ${tr('Metal 锁页不可用', 'Metal memory wiring unavailable')}`} data-model-id={item.id}>
            <i style={{ backgroundColor: 'var(--bar-fill)' }} />{item.model}{item.state === 'unloading' && ` · ${tr('卸载中', 'Unloading')}`}
            <ModelVendorMark name={item.model} size={18} />
          </span>
        ))}
        {!loaded.length && <span>{tr('暂无已加载模型', 'No loaded models')}</span>}
      </div>
      {loaded.some((item) => item.state === 'unloading') && <p className="memory-tier-notice">
        {tr('卸载中的模型保留上次上报的占用，确认释放后移除。',
          'Unloading models retain their last reported allocation until release is confirmed.')}
      </p>}
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
          const used = !reported && unknown ? tr('明细未上报', 'Breakdown not reported')
            : `${unknown ? '≥ ' : ''}${resourceBytes(total)}`;
          return (
            <div className="memory-tier" key={tier.id} data-tier={tier.id}>
              <div className="memory-tier-heading">
                <div><strong>{tier.title}</strong><small>{tier.detail}</small></div>
                <span className={unknown ? 'memory-tier-unavailable' : undefined}>
                  {resident ? `${used} / ${capacity == null ? '--' : resourceBytes(capacity)}` : used}
                </span>
              </div>
              <div className="memory-tier-track" aria-label={tier.title}
                title={tier.id === 'ssd-kv' ? separate ? tr('RAM 流式 KV 经 LRU 淘汰后的 SSD 块载荷；不含前缀缓存。', 'SSD payload of LRU-evicted RAM KV; excludes prefix caches.') : tr('当前流式 KV 的 SSD 块载荷，包含待写入块；不含持久化前缀缓存。',
                  'SSD block payload for current streamed KV, including pending writes; excludes persistent prefix cache.') : undefined}>
                {amounts.filter(({ bytes }) => scale > 0 && bytes != null && bytes > 0).map(({ item, bytes }) => (
                  <span key={item.id} data-model-id={item.id}
                    style={{ backgroundColor: 'var(--bar-fill)', width: `${bytes! / scale * 100}%` }}
                    title={`${item.model} · ${resourceBytes(bytes!)}${unknown || !capacity ? '' : ` · ${formatNumber(bytes! / capacity * 100, 1)}%`}`} />
                ))}
              </div>
              {tier.id === 'kv' && <small className="memory-tier-prefix-quota">
                {tr('其中前缀缓存', 'Prefix cache within this tier')}{' '}
                {loaded.every((item) => item.memory?.prefix_cache_bytes != null)
                  ? resourceBytes(loaded.reduce((sum, item) => sum + item.memory!.prefix_cache_bytes!, 0)) : '--'}
                {' / '}{loaded.every((item) => item.memory?.prefix_cache_limit_bytes != null)
                  ? resourceBytes(loaded.reduce((sum, item) => sum + item.memory!.prefix_cache_limit_bytes!, 0)) : '--'}
              </small>}
              {tier.id === 'ssd-kv' && !detailed && loaded.some((item) => item.memory?.ssd_kv === true) &&
                streamingStats(loaded.filter((item) => item.memory?.ssd_kv !== false))}
              {detailed && amounts.length > 0 && <div className="resource-tier-details">
                {amounts.map(({ item, bytes }) => {
                  const memory = item.memory;
                  const prefix = memory?.prefix_cache_bytes;
                  const liveKv = bytes != null && prefix != null && bytes >= prefix ? bytes - prefix : null;
                  const detail = tier.id === 'weights'
                    ? [item.devices.join(' + ').toUpperCase(), item.context_size != null ? `${formatNumber(item.context_size)} ctx` : null,
                        memory?.wired_available === true ? `${tr('锁页', 'Wired')} ${memory.wired_bytes == null ? '--' : resourceBytes(memory.wired_bytes)}`
                          : memory?.wired_available === false ? tr('锁页不可用', 'Wiring unavailable') : null].filter(Boolean).join(' · ')
                    : tier.id === 'kv'
                      ? `${tr('活跃 KV', 'Active KV')} ${liveKv == null ? '--' : resourceBytes(liveKv)} · ${separate ? tr('前缀显存', 'Prefix VRAM') : tr('前缀 RAM', 'Prefix RAM')} ${prefix == null ? '--' : resourceBytes(prefix)} / ${memory?.prefix_cache_limit_bytes == null ? '--' : resourceBytes(memory.prefix_cache_limit_bytes)}`
                      : tier.id === 'ram-experts'
                        ? memory?.ram_experts == null ? tr('未上报', 'Not reported') : memory.ram_experts ? tr('按需从 RAM 传入显存', 'Transferred from RAM to VRAM on demand') : tr('无 RAM 流式专家', 'No RAM-streamed experts')
                        : tier.id === 'ram-kv'
                          ? memory?.ram_kv == null ? tr('未上报', 'Not reported') : memory.ram_kv ? `${tr('RAM 上限', 'RAM limit')} ${memory.ram_kv_limit_bytes == null ? '--' : resourceBytes(memory.ram_kv_limit_bytes)} · ${tr('LRU 淘汰后写入 SSD', 'LRU eviction spills to SSD')}` : tr('未启用 RAM 流式 KV', 'RAM-streamed KV disabled')
                      : tier.id === 'experts'
                        ? memory?.ssd_experts == null ? tr('未上报', 'Not reported') : memory.ssd_experts ? tr('按需从 SSD 加载', 'Loaded from SSD on demand')
                          : separate ? tr('当前设备尚未支持 SSD 专家流式加载', 'SSD expert streaming is not yet supported on this device') : tr('专家全部常驻', 'All experts resident')
                        : tier.id === 'ple'
                          ? memory?.ssd_ple == null ? tr('未上报', 'Not reported') : memory.ssd_ple ? tr('按行从 SSD 读取', 'Rows read from SSD') : tr('无 SSD PLE 表', 'No SSD PLE table')
                          : memory?.ssd_kv == null ? tr('未上报', 'Not reported') : memory.ssd_kv
                            ? `${tr('常驻', 'Resident')} ${memory.streaming_kv_resident_bytes == null ? '--' : resourceBytes(memory.streaming_kv_resident_bytes)} / ${memory.streaming_kv_budget_bytes == null ? '--' : resourceBytes(memory.streaming_kv_budget_bytes)} · ${tr('待写入', 'Pending write')} ${memory.streaming_kv_pending_bytes == null ? '--' : resourceBytes(memory.streaming_kv_pending_bytes)}`
                            : tr('未启用流式稀疏注意力', 'Streaming sparse attention disabled');
                  return <div className="resource-tier-model" key={item.id} data-model-id={item.id}>
                    <div className="resource-tier-model-name"><i style={{ backgroundColor: 'var(--bar-fill)' }} />
                      <span title={item.model}>{item.model}</span></div>
                    <strong>{bytes == null ? tr('未上报', 'Not reported') : resourceBytes(bytes)}</strong>
                    <small>{detail}{tier.id === 'kv' && ` · ${memory?.context_count == null ? '--' : formatNumber(memory.context_count)} ${tr('组上下文', 'contexts')} · ${memory?.prefix_cache_blocks == null ? '--' : formatNumber(memory.prefix_cache_blocks)} ${separate ? tr('个显存缓存块', 'VRAM cache blocks') : tr('个 RAM 缓存块', 'RAM cache blocks')}`}</small>
                    {tier.id === 'ssd-kv' && memory?.ssd_kv && streamingStats([item])}
                  </div>;
                })}
              </div>}
              {tier.id === 'weights' && loaded.some((item) => item.memory?.wired_available === false) &&
                <small className="memory-tier-notice"><Icon name="info" size={12} />
                  {tr('部分模型的 Metal 锁页不可用，权重可能被系统换出。',
                    'Metal memory wiring is unavailable for some models; weights may be paged out.')}
                </small>}
              {unknown && <small className="memory-tier-notice"><Icon name="info" size={12} />
                {tr(`${missing.map(({ item }) => item.model).join('、')}：明细未上报`,
                  `${missing.map(({ item }) => item.model).join(', ')}: breakdown not reported`)}
              </small>}
            </div>
          );
        })}
      </div>
    </TMPanel>
  );
}
