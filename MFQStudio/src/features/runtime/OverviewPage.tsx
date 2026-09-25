/** 概览页负责展示运行快照、模型选择和端点复制。 */
import { useEffect, useRef, useState } from 'react';
import { useRuntime } from '../../app/RuntimeProvider';
import { useSettings } from '../settings/SettingsProvider';
import {
  Icon,
  ScreenHeader,
  SectionLabel,
  TMPanel,
  ModelMonogram,
  MetricTile,
  UsageBar,
  EmptyPanel,
} from '../../app/display';
import { errorMessage, formatNumber, formatBytes, formatDuration } from '../../app/formatters';
import { runtimeModelNames } from './modelSelection';
import { displayPrefillMetric, preferPositiveMetric } from './metrics';
import { RuntimeHero } from './RuntimeHero';
import { toast } from '../../stores/toastStore';

/** 展示共享运行状态，页面卸载时清理复制提示计时器。 */
export function OverviewPage() {
  const {
    runtime,
    models,
    instances,
    studio,
    selectedModel: model,
    setSelectedModel: selectModel,
    refreshRuntime,
    loading: busy,
  } = useRuntime();
  const { tr } = useSettings();
  const [endpointCopied, setEndpointCopied] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(
    () => () => {
      if (timer.current) clearTimeout(timer.current);
    },
    [],
  );
  const availableModelNames = runtimeModelNames(models, instances);
  const last = runtime?.last_request;
  const lastPrefill = displayPrefillMetric(last);
  const lastTtftMs = preferPositiveMetric(last?.ttft_ms, last?.complete_prefill_ms);
  /** 复制当前服务地址，并在页面仍挂载时短暂展示成功状态。 */
  async function copyEndpoint() {
    try {
      await navigator.clipboard.writeText(studio?.service_url || 'http://127.0.0.1:8090');
      setEndpointCopied(true);
      toast.success(tr('服务地址已复制到剪贴板', 'Endpoint URL copied to clipboard'));
      if (timer.current) clearTimeout(timer.current);
      timer.current = setTimeout(() => setEndpointCopied(false), 1600);
    } catch (cause) {
      toast.error(errorMessage(cause));
    }
  }
  const runtimeMemory = Number(
    runtime?.mlx_active_bytes ??
      runtime?.cuda_allocated_bytes ??
      runtime?.process_resident_bytes ??
      0,
  );
  const runtimeCache = Number(runtime?.mlx_cache_bytes ?? runtime?.cuda_reserved_bytes ?? 0);
  const runtimeMemoryCapacity =
    Number(runtime?.device_total_bytes || 0) || Math.max(runtimeMemory + runtimeCache, 1);
  const runtimeDeviceFree = Number(runtime?.device_free_bytes || 0);
  const prefixCacheQueries = Number(runtime?.prefix_cache_queries || 0);
  const prefixCacheHits = Number(runtime?.prefix_cache_hits || 0);
  const prefixCacheHitRate =
    prefixCacheQueries > 0 ? (prefixCacheHits / prefixCacheQueries) * 100 : 0;

  return (
    <section className="dashboard-view" id="dashboard-overview">
      <ScreenHeader
        title={tr('概览', 'Overview')}
        subtitle={tr('本机推理状态与性能。', 'Local inference status and performance.')}
        trailing={
          <button disabled={busy} onClick={() => void refreshRuntime()} type="button">
            <Icon name="refresh" size={14} />
            {tr('刷新', 'Refresh')}
          </button>
        }
      />
      <RuntimeHero />
      {availableModelNames.length > 1 && (
        <>
          <SectionLabel
            title={tr('已加载模型', 'Loaded models')}
            subtitle={tr(
              `${availableModelNames.length} 个可用于推理`,
              `${availableModelNames.length} available for inference`,
            )}
          />
          <TMPanel className="overview-models-panel">
            <div className="overview-model-grid">
              {availableModelNames.map((name) => {
                const instance = instances.find(
                  (candidate) => candidate.model === name && candidate.state !== 'failed',
                );
                const selected = name === model;
                const stateLabel =
                  instance?.state === 'busy' ? tr('使用中', 'Busy') : tr('就绪', 'Ready');
                const details = [
                  instance?.devices.join(' + '),
                  instance?.context_size ? `${formatNumber(instance.context_size)} ctx` : null,
                  instance
                    ? tr(
                        `${instance.active_sessions} 个会话`,
                        `${instance.active_sessions} sessions`,
                      )
                    : null,
                ]
                  .filter(Boolean)
                  .join(' · ');
                return (
                  <button
                    aria-pressed={selected}
                    className={`overview-model-card${selected ? ' selected' : ''}`}
                    disabled={busy || selected}
                    key={name}
                    onClick={() => selectModel(name)}
                    type="button"
                  >
                    <ModelMonogram name={name} state="ready" />
                    <span className="overview-model-copy">
                      <strong title={name}>{name}</strong>
                      <small>{details || stateLabel}</small>
                    </span>
                    <span className="runtime-status-pill ready">
                      <i />
                      {selected ? tr('当前', 'Current') : stateLabel}
                    </span>
                  </button>
                );
              })}
            </div>
          </TMPanel>
        </>
      )}
      <SectionLabel
        title={tr('实时性能', 'Live performance')}
        subtitle={tr(
          '最近请求吞吐与累计缓存复用',
          'Latest request throughput · cumulative cache reuse',
        )}
      />
      <div className="metric-grid">
        <MetricTile
          label={tr('预填充', 'Prefill')}
          value={`${formatNumber(lastPrefill.tokensPerSecond, 1)} tok/s`}
          detail={`${formatNumber(lastPrefill.milliseconds, 1)} ms · ${tr('输入处理', 'Prompt processing')}`}
          icon="text-forward"
        />
        <MetricTile
          label={tr('解码', 'Decode')}
          value={`${formatNumber(last?.decode_tps, 1)} tok/s`}
          detail={tr('输出生成', 'Token generation')}
          icon="waveform"
        />
        <MetricTile
          label={tr('首字延迟', 'TTFT')}
          value={`${formatNumber(lastTtftMs, 1)} ms`}
          detail={tr('首次输出耗时', 'Time to first token')}
          icon="clock"
        />
        <MetricTile
          label={tr('前缀复用', 'Prefix reuse')}
          value={prefixCacheQueries > 0 ? `${formatNumber(prefixCacheHitRate, 1)}%` : '--'}
          detail={tr(
            `已恢复 ${formatNumber(runtime?.prefix_cache_hit_tokens || 0)} tokens`,
            `${formatNumber(runtime?.prefix_cache_hit_tokens || 0)} tokens restored`,
          )}
          icon="reuse"
        />
        <MetricTile
          label={tr('内存', 'Memory')}
          value={formatBytes(runtimeMemory)}
          detail={
            runtimeCache
              ? tr(
                  `分配器缓存 ${formatBytes(runtimeCache)}`,
                  `${formatBytes(runtimeCache)} allocator cache`,
                )
              : tr('模型驻留', 'Runtime residency')
          }
          icon="memory"
        />
      </div>
      <SectionLabel title={tr('内存层级', 'Memory hierarchy')} />
      {runtime ? (
        <TMPanel className="overview-memory-panel">
          <div className="overview-memory-heading">
            <div>
              <h2>{tr('推理内存', 'Runtime memory')}</h2>
              <p>
                {tr(
                  '模型驻留、分配器缓存与可复用前缀共享统一内存。',
                  'Model residency, allocator cache, and reusable prefixes share unified memory.',
                )}
              </p>
            </div>
            <strong>{formatBytes(runtimeMemory + runtimeCache)}</strong>
          </div>
          <div className="overview-memory-bars">
            <UsageBar
              label={tr('模型与活动张量', 'Model and active tensors')}
              used={runtimeMemory}
              total={runtimeMemoryCapacity}
            />
            <UsageBar
              label={tr('分配器缓存', 'Allocator cache')}
              used={runtimeCache}
              total={runtimeMemoryCapacity}
            />
          </div>
          <div className="overview-memory-facts">
            <span>
              <Icon name="memory" size={13} />
              {formatBytes(runtimeDeviceFree)} {tr('设备可用', 'device free')}
            </span>
            <span>
              <Icon name="reuse" size={13} />
              {formatNumber(runtime?.prefix_cache_snapshots || 0)}{' '}
              {tr('个前缀快照', 'prefix snapshots')}
            </span>
            <span>
              <Icon name="text-forward" size={13} />
              {tr(
                `已复用 ${formatNumber(runtime?.prefix_cache_hit_tokens || 0)} tokens`,
                `${formatNumber(runtime?.prefix_cache_hit_tokens || 0)} tokens reused`,
              )}
            </span>
            <span>
              <Icon name="queue" size={13} />
              {formatNumber(runtime?.active_requests || 0)} {tr('个活动请求', 'active requests')}
            </span>
          </div>
        </TMPanel>
      ) : (
        <EmptyPanel
          icon="memory"
          title={tr('推理内存尚未上报', 'Runtime memory is unavailable')}
          message={tr(
            '服务器就绪后会显示模型驻留与缓存状态。',
            'Memory residency and cache state appear when the server is ready.',
          )}
        />
      )}
      <div className="overview-footer-grid">
        <TMPanel className="overview-endpoint-panel">
          <div className="overview-panel-title">
            <Icon name="link" size={15} />
            <h2>{tr('OpenAI 兼容端点', 'OpenAI-compatible endpoint')}</h2>
            <button
              aria-label={endpointCopied ? tr('已复制', 'Copied') : tr('复制端点', 'Copy endpoint')}
              className={endpointCopied ? 'copied' : ''}
              onClick={() => void copyEndpoint()}
              title={endpointCopied ? tr('已复制', 'Copied') : tr('复制端点', 'Copy endpoint')}
              type="button"
            >
              <Icon name={endpointCopied ? 'check' : 'copy'} size={14} />
            </button>
          </div>
          <code>{studio?.service_url || 'http://127.0.0.1:8090'}</code>
          <p>
            {tr(
              '可直接用于 OpenAI SDK；默认回环地址不经过云端。',
              'Use this base URL with OpenAI SDKs. The default loopback address sends no traffic to the cloud.',
            )}
          </p>
        </TMPanel>
        <TMPanel className="overview-session-panel">
          <div className="overview-panel-title">
            <Icon name="chart" size={15} />
            <h2>{tr('会话统计', 'Session')}</h2>
            <span>{formatDuration(runtime?.uptime_seconds)}</span>
          </div>
          <div className="overview-session-stats">
            <div>
              <strong>{formatNumber(runtime?.total_requests || 0)}</strong>
              <small>{tr('已完成', 'Completed')}</small>
            </div>
            <div>
              <strong>{formatNumber(runtime?.total_prompt_tokens || 0)}</strong>
              <small>{tr('提示词', 'Prompt')}</small>
            </div>
            <div>
              <strong>{formatNumber(runtime?.total_completion_tokens || 0)}</strong>
              <small>{tr('已生成', 'Generated')}</small>
            </div>
          </div>
          <p>
            {formatNumber(runtime?.failed_requests || 0)} {tr('个失败请求', 'failed requests')} ·{' '}
            {formatNumber(runtime?.active_requests || 0)} {tr('个活动请求', 'active')}
          </p>
        </TMPanel>
      </div>
    </section>
  );
}
