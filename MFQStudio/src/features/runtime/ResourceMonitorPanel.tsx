/** Provide ResourceMonitorPanel interface behavior. */
import { useTranslation } from 'react-i18next';
import { useEffect, useState } from 'react';
import { useRuntime } from '../../app/RuntimeProvider';
import { SectionLabel, TMPanel } from '../../app/display';
import { errorMessage, formatNumber } from '../../app/formatters';
import { runtimeApi } from '../../shared/api/resources/runtime';
import type { RuntimeResources } from '../../shared/api/types';

function rate(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return '--';
  return value >= 2 ** 30 ? `${formatNumber(value / 2 ** 30, 2)} GiB/s`
    : `${formatNumber(value / 2 ** 20, 2)} MiB/s`;
}

/** Subscribe to runtime resource samples and render compute and bandwidth utilization. */
export function ResourceMonitorPanel() {
  const { ready, connectionRevision } = useRuntime();
  const { t } = useTranslation();
  const [resources, setResources] = useState<RuntimeResources | null>(null);
  const [error, setError] = useState('');
  useEffect(() => {
    let disposed = false;
    let timer: ReturnType<typeof setTimeout>;
    setResources(null);
    setError('');
    async function poll() {
      try {
        const next = await runtimeApi.runtimeResources();
        if (!disposed) { setResources(next); setError(''); }
      } catch (cause) {
        if (!disposed) { setResources(null); setError(errorMessage(cause)); }
      } finally {
        if (!disposed) timer = setTimeout(() => void poll(), 2000);
      }
    }
    if (ready) void poll();
    return () => { disposed = true; clearTimeout(timer); };
  }, [ready, connectionRevision]);
  const unavailable = t('runtime:resourceMonitorPanel.notReported');
  function utilization(value: number | null | undefined) {
    return value == null ? unavailable : `${formatNumber(value, 1)}%`;
  }
  function deviceLabel(kind: string, name?: string | null, cores?: number | null) {
    return `${kind}${t('runtime:resourceMonitorPanel.message')}${name && name !== kind ? name : unavailable}${cores != null ? ` ${cores}C` : ''}`;
  }
  const compute = [{ name: deviceLabel('CPU', resources?.cpu_name, resources?.cpu_cores),
    utilization_percent: resources?.cpu_utilization_percent },
    ...(resources?.gpus.length ? resources.gpus.map((gpu) => ({
      name: deviceLabel('GPU', gpu.name, gpu.core_count), utilization_percent: gpu.utilization_percent,
    })) : [{ name: deviceLabel('GPU'), utilization_percent: null }])];
  return <>
    <SectionLabel title={t('runtime:resourceMonitorPanel.resourceMonitoring')}
      subtitle={t('runtime:resourceMonitorPanel.serverDeviceSampledEvery2Seconds')} />
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
          <small>{t('runtime:resourceMonitorPanel.systemUtilization')}</small>
        </div>)}
        <div className="resource-monitor-metric">
          <span>{t('runtime:resourceMonitorPanel.memoryBandwidth')}</span>
          <strong>{rate(resources?.memory_bandwidth_bytes_per_second)}</strong>
          <small>{t('runtime:resourceMonitorPanel.utilization')} {utilization(resources?.memory_bandwidth_utilization_percent)}</small>
          <small>{resources?.memory_bandwidth_limit_bytes_per_second
            ? `${t('runtime:resourceMonitorPanel.specifiedLimit')} ${rate(resources.memory_bandwidth_limit_bytes_per_second)}`
            : t('runtime:resourceMonitorPanel.noSystemBandwidthCounter')}</small>
        </div>
      </div>
      <div className="resource-monitor-section">
        <h2>{t('runtime:resourceMonitorPanel.ssdDiskTraffic')}</h2>
        <div className="resource-traffic-grid">
          <span>{t('runtime:resourceMonitorPanel.device')}</span><span>{t('runtime:resourceMonitorPanel.read')}</span><span>{t('runtime:resourceMonitorPanel.write')}</span><span>{t('runtime:resourceMonitorPanel.bandwidthUtilization')}</span>
          {(resources?.disks.length ? resources.disks : [{ name: 'SSD', read_bytes_per_second: null, write_bytes_per_second: null, bandwidth_utilization_percent: null }]).map((disk) =>
            <div className="resource-traffic-row" key={disk.name}>
              <strong>{disk.name}</strong><span>{rate(disk.read_bytes_per_second)}</span>
              <span>{rate(disk.write_bytes_per_second)}</span><span>{utilization(disk.bandwidth_utilization_percent)}</span>
            </div>)}
        </div>
      </div>
      <div className="resource-monitor-section">
        <h2>{t('runtime:resourceMonitorPanel.weightTraffic')}</h2>
        <p>{t('runtime:resourceMonitorPanel.logicalFileReadsIncludeSystemPageCacheHitsNotPhysicalSsdOr')}</p>
        {resources?.weights.length ? resources.weights.map((weight) => <div className="resource-weight-row" key={weight.instance_id}>
          <strong title={weight.model}>{weight.model}</strong>
          <div><span>{t('runtime:resourceMonitorPanel.streamedExperts')}</span><b>{rate(weight.expert_read_bytes_per_second)}</b></div>
          <div><span>PLE</span><b>{rate(weight.ple_read_bytes_per_second)}</b></div>
          {weight.engram_read_bytes_per_second != null && <div><span>Engram</span><b>{rate(weight.engram_read_bytes_per_second)}</b></div>}
        </div>) : <p>{t('runtime:resourceMonitorPanel.loadAModelToMonitorPerModelWeightReads')}</p>}
        <p>{t('runtime:resourceMonitorPanel.residentWeightReadBandwidthNotReported')}</p>
      </div>
    </TMPanel>
  </>;
}
