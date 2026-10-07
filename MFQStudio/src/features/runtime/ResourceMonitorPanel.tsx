import { useEffect, useState } from 'react';
import { useRuntime } from '../../app/RuntimeProvider';
import { SectionLabel, TMPanel } from '../../app/display';
import { errorMessage, formatNumber } from '../../app/formatters';
import { useSettings } from '../settings/SettingsProvider';
import { runtimeApi } from '../../shared/api/resources/runtime';
import type { RuntimeResources } from '../../shared/api/types';

function rate(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return '--';
  return value >= 2 ** 30 ? `${formatNumber(value / 2 ** 30, 2)} GiB/s`
    : `${formatNumber(value / 2 ** 20, 2)} MiB/s`;
}

export function ResourceMonitorPanel() {
  const { ready, connectionRevision } = useRuntime();
  const { tr } = useSettings();
  const [resources, setResources] = useState<RuntimeResources | null>(null);
  const [error, setError] = useState('');
  useEffect(() => {
    let disposed = false;
    let polling = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const controller = new AbortController();
    setResources(null);
    setError('');
    async function poll() {
      if (disposed || polling) return;
      clearTimeout(timer);
      if (document.hidden) {
        timer = setTimeout(() => void poll(), 2000);
        return;
      }
      polling = true;
      try {
        const next = await runtimeApi.runtimeResources(controller.signal);
        if (!disposed) { setResources(next); setError(''); }
      } catch (cause) {
        if (!disposed) { setResources(null); setError(errorMessage(cause)); }
      } finally {
        polling = false;
        if (!disposed) timer = setTimeout(() => void poll(), 2000);
      }
    }
    function visible() { if (!document.hidden) void poll(); }
    if (ready) {
      document.addEventListener('visibilitychange', visible);
      void poll();
    }
    return () => { disposed = true; controller.abort(); clearTimeout(timer); document.removeEventListener('visibilitychange', visible); };
  }, [ready, connectionRevision]);
  const unavailable = tr('未上报', 'Not reported');
  function utilization(value: number | null | undefined) {
    return value == null ? unavailable : `${formatNumber(value, 1)}%`;
  }
  function deviceLabel(kind: string, name?: string | null, cores?: number | null) {
    return `${kind}${tr('：', ': ')}${name && name !== kind ? name : unavailable}${cores != null ? ` ${cores}C` : ''}`;
  }
  const compute = [{ name: deviceLabel('CPU', resources?.cpu_name, resources?.cpu_cores),
    utilization_percent: resources?.cpu_utilization_percent },
    ...(resources?.gpus.length ? resources.gpus.map((gpu) => ({
      name: deviceLabel('GPU', gpu.name, gpu.core_count), utilization_percent: gpu.utilization_percent,
    })) : [{ name: deviceLabel('GPU'), utilization_percent: null }])];
  return <>
    <SectionLabel title={tr('资源监控', 'Resource monitoring')}
      subtitle={tr('服务所在设备 · 每 2 秒采样', 'Server device · sampled every 2 seconds')} />
    <TMPanel className="resource-monitor-panel">
      {error && <p className="resource-monitor-error" role="status">{error}</p>}
      <div className="resource-monitor-grid">
        {compute.map((item, index) => <div className="resource-monitor-metric" key={`${item.name}-${index}`}>
          <span>{item.name}</span><strong>{utilization(item.utilization_percent)}</strong>
          <div className="resource-utilization-track" role="meter" aria-label={item.name}
            aria-valuemin={0} aria-valuemax={100} aria-valuenow={item.utilization_percent ?? undefined}
            aria-valuetext={utilization(item.utilization_percent)}>
            {item.utilization_percent != null && <i style={{ width: `${item.utilization_percent}%` }} />}
          </div>
          <small>{tr('整机占用', 'System utilization')}</small>
        </div>)}
        <div className="resource-monitor-metric">
          <span>{tr('内存带宽', 'Memory bandwidth')}</span>
          <strong>{rate(resources?.memory_bandwidth_bytes_per_second)}</strong>
          <small>{tr('占用率', 'Utilization')} {utilization(resources?.memory_bandwidth_utilization_percent)}</small>
          <small>{resources?.memory_bandwidth_limit_bytes_per_second
            ? `${tr('规格上限', 'Specified limit')} ${rate(resources.memory_bandwidth_limit_bytes_per_second)}`
            : tr('系统未提供带宽计数器', 'No system bandwidth counter')}</small>
        </div>
      </div>
      <div className="resource-monitor-section">
        <h2>{tr('SSD / 磁盘流量', 'SSD / disk traffic')}</h2>
        <div className="resource-traffic-grid">
          <span>{tr('设备', 'Device')}</span><span>{tr('读取', 'Read')}</span><span>{tr('写入', 'Write')}</span><span>{tr('带宽占用率', 'Bandwidth utilization')}</span>
          {(resources?.disks.length ? resources.disks : [{ name: 'SSD', read_bytes_per_second: null, write_bytes_per_second: null, bandwidth_utilization_percent: null }]).map((disk) =>
            <div className="resource-traffic-row" key={disk.name}>
              <strong>{disk.name}</strong><span>{rate(disk.read_bytes_per_second)}</span>
              <span>{rate(disk.write_bytes_per_second)}</span><span>{utilization(disk.bandwidth_utilization_percent)}</span>
            </div>)}
        </div>
      </div>
      <div className="resource-monitor-section">
        <h2>{tr('权重流量', 'Weight traffic')}</h2>
        <p>{tr('文件逻辑读取；系统页缓存命中也计入，不等于物理 SSD 或内存总线流量。', 'Logical file reads include system page-cache hits, not physical SSD or memory-bus traffic.')}</p>
        {resources?.weights.length ? resources.weights.map((weight) => <div className="resource-weight-row" key={weight.instance_id}>
          <strong title={weight.model}>{weight.model}</strong>
          <div><span>{tr('流式专家', 'Streamed experts')}</span><b>{rate(weight.expert_read_bytes_per_second)}</b></div>
          <div><span>PLE</span><b>{rate(weight.ple_read_bytes_per_second)}</b></div>
          {weight.engram_read_bytes_per_second != null && <div><span>Engram</span><b>{rate(weight.engram_read_bytes_per_second)}</b></div>}
        </div>) : <p>{tr('加载模型后显示逐模型权重读取。', 'Load a model to monitor per-model weight reads.')}</p>}
        <p>{tr('常驻权重读取带宽：未上报', 'Resident weight read bandwidth: not reported')}</p>
      </div>
    </TMPanel>
  </>;
}
