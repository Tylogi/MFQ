import { memo, useRef, type ReactNode } from 'react';
import { QuestionIcon } from '@phosphor-icons/react';
import type { RuntimeInstance, RuntimeStatus } from '../../shared/api/types';
import { formatBytes, formatNumber } from '../../app/formatters';
import { Icon, TMPanel } from '../../app/display';
import { useSettings } from '../settings/SettingsProvider';
import { ModelVendorMark } from '../../app/ModelVendorMark';
import { Tooltip } from '../../shared/ui/Tooltip';
import { hasSeparateVram, memoryPoolUsage, residentWeightsAndBuffers, gpuMemoryGroups } from './memoryArchitecture';

type Tier = 'weights' | 'kv' | 'ram-experts' | 'ram-kv' | 'ram-prefix' | 'experts' | 'ple' | 'ssd-kv';
const modelCount = (count: number) => `${count} ${count === 1 ? 'model' : 'models'}`;
const resourceBytes = (bytes: number) => bytes < 1024 ? `${formatNumber(bytes)} B` :
  formatBytes(bytes).replace(/\b(KB|MB|GB|TB)\b/g, (unit) => `${unit[0]}iB`);

// Rendered inside the overview page, which re-renders on the 500 ms serving
// poll. The memory picture only changes when instances or runtime do.
export const MemoryHierarchy = memo(function MemoryHierarchy({
  instances, memoryCapacityBytes, runtime, connectionRevision = 0, detailed = false, summary,
}: {
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
    if (tier === 'weights') return residentWeightsAndBuffers(memory, separate);
    if (tier === 'kv') return memory.kv_bytes;
    if (tier === 'ram-experts') return memory.ram_experts === false ? 0 : memory.ram_expert_bytes ?? null;
    if (tier === 'ram-kv') return memory.ram_kv === false ? 0 : memory.ram_kv_bytes ?? null;
    if (tier === 'ram-prefix') return memory.prefix_cache_ram_bytes ?? 0;
    if (tier === 'experts') return memory.ssd_experts === false ? 0 : memory.ssd_expert_bytes;
    if (tier === 'ssd-kv') return memory.ssd_kv === false ? 0 : memory.ssd_kv_bytes ?? null;
    return memory.ssd_ple === false ? 0 : memory.ssd_ple_bytes;
  };
  const pool = (capacity: number | null | undefined, weights: Tier, caches: Tier, available?: number | null) =>
    memoryPoolUsage(capacity, loaded.map(item => value(item, weights)), loaded.map(item => value(item, caches)), available);
  const residentPool = pool(memoryCapacityBytes, 'weights', 'kv',
    separate ? runtime?.device_free_bytes : runtime?.runtime_memory_effective_budget_bytes != null ? undefined : runtime?.host_memory_available_bytes);
  const ramPool = separate ? memoryPoolUsage(runtime?.host_memory_total_bytes,
    loaded.map(item => value(item, 'ram-experts')), loaded.map(item => {
      const kv = value(item, 'ram-kv');
      return kv == null ? null : kv + (value(item, 'ram-prefix') ?? 0);
    }), runtime?.host_memory_available_bytes) : residentPool;
  const ramPrefixCapacity = ramPool.cacheCapacity == null ? null : Math.max(0,
    ramPool.cacheCapacity - loaded.reduce((sum, item) => sum + (value(item, 'ram-kv') ?? 0), 0));
  const contextCount = loaded.every((item) => item.memory?.context_count != null)
    ? formatNumber(loaded.reduce((sum, item) => sum + item.memory!.context_count!, 0)) : '--';
  const cacheBlocks = loaded.every((item) => item.memory?.prefix_cache_blocks != null)
    ? formatNumber(loaded.reduce((sum, item) => sum + item.memory!.prefix_cache_blocks!, 0)) : '--';
  const tiers: { id: Tier; title: string; detail: string }[] = [
    { id: 'weights', title: separate ? tr('显存常驻权重和缓冲区', 'VRAM-resident weights and buffers') : tr('常驻专家与稠密权重', 'Resident experts and dense weights'),
      detail: tr(`${loaded.length} 个模型`, modelCount(loaded.length)) },
    { id: 'kv', title: separate ? tr('显存常驻 KV Cache 与前缀缓存', 'VRAM-resident KV and prefix cache') : tr('常驻 KV Cache 与前缀缓存', 'Resident KV and prefix cache'),
      detail: tr(`${contextCount} 组上下文，${cacheBlocks} 个缓存块`, `${contextCount} contexts · ${cacheBlocks} cache blocks`) },
    ...(separate ? [
      { id: 'ram-experts' as const, title: tr('内存流式加载专家', 'RAM-streamed experts'), detail: streamedCount('ram_experts') },
      { id: 'ram-kv' as const, title: tr('内存流式 KV', 'RAM-streamed KV'), detail: streamedCount('ram_kv') },
      { id: 'ram-prefix' as const, title: tr('RAM 前缀缓存', 'RAM prefix cache'),
        detail: tr('低优先级，内存紧张时按 LRU 释放', 'Low priority; reclaimed by LRU under memory pressure') },
    ] : []),
    { id: 'experts', title: tr('SSD 流式加载专家', 'SSD-streamed experts'), detail: separate && loaded.length && loaded.every(item => item.memory?.ssd_experts === false)
      ? tr('当前设备尚未支持', 'Not yet supported on this device') : streamedCount('ssd_experts') },
    { id: 'ple', title: tr('SSD 流式加载 PLE 表', 'SSD-streamed PLE tables'), detail: streamedCount('ssd_ple') },
    { id: 'ssd-kv', title: tr('SSD 流式 KV', 'SSD-streamed KV'), detail: streamedCount('ssd_kv') },
  ];
  const gpuGroups = separate ? gpuMemoryGroups(runtime, loaded) : [];
  const groups: { name: string; ids: Tier[]; detail: string; gpu?: typeof gpuGroups[number] }[] = [
    ...gpuGroups.map(gpu => ({ name: 'VRAM', ids: ['weights', 'kv'] as Tier[], gpu,
      detail: gpu.name || tr('常驻权重与热缓存', 'Resident weights and hot caches') })),
    { name: 'RAM', ids: separate ? ['ram-experts', 'ram-kv', 'ram-prefix'] : ['weights', 'kv'],
      detail: separate ? tr('流式专家、KV 与前缀缓存', 'Streamed experts, KV and prefix cache') : tr('常驻权重与热缓存', 'Resident weights and hot caches') },
    { name: 'SSD', ids: ['experts', 'ple', 'ssd-kv'],
      detail: tr('按需加载的权重与 KV', 'Weights and KV loaded on demand') },
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
          <div className="memory-heading-title">
            <h2>{detailed ? tr('资源分配', 'Resource allocation') : tr('运行资源', 'Runtime resources')}</h2>
            <Tooltip content={<div>
            <strong>{tr('资源分工', 'Resource tiers')}</strong>
            <p>{separate ? tr('VRAM：常驻专家与稠密权重、活跃 KV 和前缀热缓存。', 'VRAM: resident weights, buffers, active KV and hot prefixes.')
              : tr('RAM：常驻专家与稠密权重、活跃 KV 和前缀热缓存；统一内存设备由 CPU 与 GPU 共享。', 'RAM: resident experts and dense weights, active KV and hot prefixes; unified-memory devices share it between CPU and GPU.')}</p>
            {separate && <p>{tr('RAM：流式专家和 KV 的中间层，按需传入显存；冷 KV 经 LRU 淘汰后写入 SSD。', 'RAM: an intermediate tier for streamed experts and KV, transferred to VRAM on demand; LRU-evicted cold KV is written to SSD.')}</p>}
            <p>{tr('SSD：流式专家按需加载，PLE 表按 token 读取对应行，流式 KV 按选中的块读取。', 'SSD: experts are loaded on demand, PLE tables supply the rows for each token, and streamed KV supplies selected blocks.')}</p>
            <p>{separate ? tr('前缀缓存使用显存、RAM 和可选 SSD 三层。RAM 优先让出内存，按 LRU 转存至 SSD；关闭 SSD 时直接销毁。', 'Prefix caching uses VRAM, RAM and optional SSD. RAM yields to other allocations: LRU eviction spills to SSD, or discards blocks when SSD is disabled.') : tr('前缀缓存由 RAM 热层与 SSD 冷层组成。', 'Prefix caching uses a RAM hot tier and SSD cold tier.')}</p>
            <p>{tr('SSD 流式 KV 不含前缀缓存，后者在缓存管理中单独统计。', 'Streamed SSD KV excludes prefix caches, which are tracked separately in cache management.')}</p>
          </div>}>
              <button className="memory-capacity-help" type="button" aria-label={tr('资源分层说明', 'Memory tier details')}>
                <QuestionIcon size={16} aria-hidden="true" />
              </button>
            </Tooltip>
          </div>
          <p className="memory-capacity-note">{tr('下方内存、显存上限为扣除系统和其他进程占用后的 MFQ 可用容量，包含 MFQ 已占用与当前空余空间。',
            'Memory and VRAM limits below reflect capacity available to MFQ after system and other process usage, including MFQ allocations and remaining free space.')}</p>
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
        {groups.map((group) => {
          const groupItems = group.gpu?.instances ?? loaded;
          const groupPool = group.gpu ? memoryPoolUsage(gpuGroups.length === 1
            ? memoryCapacityBytes ?? group.gpu.memory_total_bytes : group.gpu.memory_total_bytes,
            groupItems.map(item => value(item, 'weights')), groupItems.map(item => value(item, 'kv')),
            group.gpu.memory_available_bytes) : residentPool;
          return <section className="memory-tier-group" key={group.gpu ? `gpu-${group.gpu.id}` : group.name}
            data-memory-group={group.name} data-gpu-id={group.gpu?.id} data-storage={group.name.toLowerCase()}>
          <div className="memory-tier-group-heading"><h3>{group.name === 'VRAM' ? `${tr('显存', 'VRAM')} · GPU ${group.gpu!.id}` : group.name === 'RAM' ? tr('内存', 'RAM') : group.name}</h3><i aria-hidden="true" /><span>{group.detail}</span></div>
          {tiers.filter((tier) => group.ids.includes(tier.id)).map((tier) => {
          const amounts = groupItems.map((item) => ({ item, bytes: value(item, tier.id) }));
          const missing = amounts.filter(({ bytes }) => bytes == null);
          const unknown = missing.length > 0;
          const reported = amounts.length - missing.length;
          const total = amounts.reduce((sum, { bytes }) => sum + (bytes ?? 0), 0);
          const resident = tier.id === 'weights' || tier.id === 'kv' || tier.id === 'ram-experts' || tier.id === 'ram-kv' || tier.id === 'ram-prefix';
          const capacity = tier.id === 'weights' ? groupPool.capacity : tier.id === 'kv' ? groupPool.cacheCapacity
            : tier.id === 'ram-experts' ? ramPool.capacity : tier.id === 'ram-kv' ? ramPool.cacheCapacity
              : tier.id === 'ram-prefix' ? ramPrefixCapacity : total;
          const scale = capacity == null ? 0 : Math.max(capacity, total);
          const used = !reported && unknown ? tr('明细未上报', 'Breakdown not reported')
            : `${unknown ? '≥ ' : ''}${resourceBytes(total)}`;
          return (
            <div className="memory-tier" key={tier.id} data-tier={tier.id}>
              <div className="memory-tier-heading">
                <div><strong>{tier.title}</strong><small>{group.gpu && tier.id === 'weights'
                  ? tr(`${groupItems.length} 个模型`, modelCount(groupItems.length))
                  : group.gpu && !groupItems.length && tier.id === 'kv'
                    ? tr('0 组上下文，0 个缓存块', '0 contexts · 0 cache blocks') : tier.detail}</small></div>
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
                {groupItems.every((item) => item.memory?.prefix_cache_bytes != null)
                  ? resourceBytes(groupItems.reduce((sum, item) => sum + item.memory!.prefix_cache_bytes!, 0)) : '--'}
                {gpuGroups.length <= 1 && <>{' / '}{groupItems.every((item) => item.memory?.prefix_cache_limit_bytes != null)
                  ? resourceBytes(groupItems.reduce((sum, item) => sum + item.memory!.prefix_cache_limit_bytes!, 0)) : '--'}</>}
              </small>}
              {tier.id === 'ssd-kv' && !detailed && loaded.some((item) => item.memory?.ssd_kv === true) &&
                streamingStats(loaded.filter((item) => item.memory?.ssd_kv !== false))}
              {detailed && amounts.length > 0 && <div className="resource-tier-details">
                {amounts.map(({ item, bytes }) => {
                  const memory = item.memory;
                  const prefix = memory?.prefix_cache_bytes;
                  const liveKv = bytes != null && prefix != null && bytes >= prefix ? bytes - prefix : null;
                  const detail = tier.id === 'weights'
                    ? [group.gpu ? `GPU ${group.gpu.id}` : item.devices.join(' + ').toUpperCase(),
                        separate ? `${tr('权重', 'Weights')} ${memory?.resident_weight_bytes == null ? '--' : resourceBytes(memory.resident_weight_bytes)} · ${tr('缓冲区', 'Buffers')} ${memory?.resident_buffer_bytes == null ? '--' : resourceBytes(memory.resident_buffer_bytes)}` : null,
                        item.context_size != null ? `${formatNumber(item.context_size)} ctx` : null,
                        memory?.wired_available === true ? `${tr('锁页', 'Wired')} ${memory.wired_bytes == null ? '--' : resourceBytes(memory.wired_bytes)}`
                          : memory?.wired_available === false ? tr('锁页不可用', 'Wiring unavailable') : null].filter(Boolean).join(' · ')
                    : tier.id === 'kv'
                      ? `${tr('活跃 KV', 'Active KV')} ${liveKv == null ? '--' : resourceBytes(liveKv)} · ${separate ? tr('前缀显存', 'Prefix VRAM') : tr('前缀 RAM', 'Prefix RAM')} ${prefix == null ? '--' : resourceBytes(prefix)}${gpuGroups.length > 1 ? '' : ` / ${memory?.prefix_cache_limit_bytes == null ? '--' : resourceBytes(memory.prefix_cache_limit_bytes)}`}`
                      : tier.id === 'ram-prefix'
                        ? `${tr('缓存块', 'Cache blocks')} ${formatNumber(memory?.prefix_cache_ram_blocks ?? 0)} · ${tr('配额', 'Allowance')} ${resourceBytes(memory?.prefix_cache_ram_limit_bytes ?? 0)}`
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
        </section>; })}
      </div>
    </TMPanel>
  );
});
