import { useEffect } from 'react';
import { useRuntime } from '../../app/RuntimeProvider';
import { formatBytes } from '../../app/formatters';
import { SectionLabel } from '../../app/display';
import { useSettings } from '../settings/SettingsProvider';
import { MemoryHierarchy } from './MemoryHierarchy';

export function ResourceAllocationPanel() {
  const { runtime, instances, ready, connectionRevision, refreshRuntime, refreshError } = useRuntime();
  const { tr } = useSettings();
  useEffect(() => {
    if (!ready) return;
    let disposed = false;
    let polling = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    async function poll() {
      if (disposed || polling) return;
      clearTimeout(timer);
      polling = true;
      try {
        if (!document.hidden) await refreshRuntime();
      } finally {
        polling = false;
        if (!disposed) timer = setTimeout(() => void poll(), 5000);
      }
    }
    function visible() { if (!document.hidden) void poll(); }
    document.addEventListener('visibilitychange', visible);
    void poll();
    return () => { disposed = true; clearTimeout(timer); document.removeEventListener('visibilitychange', visible); };
  }, [ready, connectionRevision, refreshRuntime]);
  const bytes = (value: number | null | undefined) => value == null || !Number.isFinite(value) ? '--'
    : value < 1024 ? `${value} B` : formatBytes(value).replace(/\b(KB|MB|GB|TB)\b/g, (unit) => `${unit[0]}iB`);
  const capacity = runtime?.runtime_memory_effective_budget_bytes ?? runtime?.runtime_memory_budget_bytes;
  const loaded = (instances || []).filter((item) => item.state === 'ready' || item.state === 'busy' || item.state === 'unloading');
  const unloading = loaded.filter((item) => item.state === 'unloading').length;
  const metrics = [
    { label: tr('总常驻内存预算', 'Total resident memory budget'), value: bytes(capacity),
      detail: runtime?.runtime_memory_budget_mode === 'automatic' ? tr('自动 · 当前可用预算', 'Automatic · current usable budget')
        : runtime?.runtime_memory_budget_mode === 'explicit' ? tr('手动上限', 'Manual limit') : tr('共享内存预算', 'Shared memory budget') },
    { label: tr('已计入预算', 'Budget committed'), value: bytes(runtime?.runtime_memory_committed_bytes),
      detail: tr('权重、活跃 KV 与前缀 RAM', 'Weights, active KV, and prefix RAM') },
    { label: tr('当前剩余可用', 'Available headroom'), value: bytes(runtime?.runtime_memory_headroom_bytes),
      detail: tr('受预算与系统可用内存共同约束', 'Limited by budget and available system memory') },
    { label: tr('已加载模型', 'Loaded models'), value: String(loaded.length),
      detail: tr(`${loaded.reduce((sum, item) => sum + item.active_sessions, 0)} 个活动会话 · ${loaded.reduce((sum, item) => sum + item.queued_requests, 0)} 个排队请求`,
        `${loaded.reduce((sum, item) => sum + item.active_sessions, 0)} active sessions · ${loaded.reduce((sum, item) => sum + item.queued_requests, 0)} queued requests`)
        + (unloading ? tr(` · ${unloading} 个卸载中`, ` · ${unloading} unloading`) : '') },
  ];
  return <>
    <SectionLabel title={tr('资源明细', 'Resource breakdown')} subtitle={tr('所有已加载模型 · 每 5 秒刷新', 'All loaded models · refreshed every 5 seconds')} />
    <MemoryHierarchy instances={instances || []} memoryCapacityBytes={capacity} connectionRevision={connectionRevision} detailed
      summary={<>{refreshError && <p className="resource-monitor-error" role="status">{refreshError}{tr(' · 显示上次获取的资源明细', ' · Showing the previous resource breakdown')}</p>}
        <div className="resource-budget-summary">{metrics.map((metric) => <div key={metric.label}>
        <span>{metric.label}</span><strong>{metric.value}</strong><small>{metric.detail}</small>
      </div>)}</div></>} />
  </>;
}
