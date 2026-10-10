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
  EmptyPanel,
} from '../../app/display';
import { errorMessage, formatNumber, formatBytes, formatDuration } from '../../app/formatters';
import { runtimeModelNames } from './modelSelection';
import { displayPrefillMetric, preferPositiveMetric } from './metrics';
import { RuntimeHero } from './RuntimeHero';
import { MemoryHierarchy } from './MemoryHierarchy';
import { anthropicEndpoint, openAIEndpoint } from './endpoint';
import { getApiBaseUrl, getApiToken } from '../../shared/api/client';
import { runtimeApi } from '../../shared/api/resources/runtime';
import { toast } from '../../stores/toastStore';
import { ModelVendorMark } from '../../app/ModelVendorMark';

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
    connectionRevision,
  } = useRuntime();
  const { tr } = useSettings();
  const [endpointCopied, setEndpointCopied] = useState<string | null>(null);
  const [listener, setListener] = useState<{ url: string; revision: number; port: number | null } | null>(null);
  const serviceUrl = studio?.service_url || getApiBaseUrl();
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(
    () => () => {
      if (timer.current) clearTimeout(timer.current);
    },
    [],
  );
  useEffect(() => {
    let disposed = false;
    let pending = false;
    const endpoint = getApiBaseUrl();
    const credential = getApiToken();
    const current = () => !disposed && endpoint === getApiBaseUrl() && credential === getApiToken();
    const refresh = async () => {
      if (pending) return;
      pending = true;
      try {
        const value = await runtimeApi.runtimeListener();
        if (current()) setListener({ url: serviceUrl, revision: connectionRevision, port: value.anthropic_port ?? null });
      } catch {
        if (current()) setListener(null);
      } finally {
        pending = false;
      }
    };
    void refresh();
    const poll = setInterval(() => void refresh(), 5000);
    window.addEventListener('focus', refresh);
    return () => { disposed = true; clearInterval(poll); window.removeEventListener('focus', refresh); };
  }, [serviceUrl, connectionRevision]);
  const availableModelNames = runtimeModelNames(models, instances);
  const last = runtime?.last_request;
  const lastPrefill = displayPrefillMetric(last);
  const lastTtftMs = preferPositiveMetric(last?.ttft_ms, last?.complete_prefill_ms);
  const anthropicPort = listener?.url === serviceUrl && listener.revision === connectionRevision ? listener.port : null;
  const endpoints = [
    { id: 'openai', label: tr('OpenAI 兼容端点', 'OpenAI-compatible endpoint'), url: openAIEndpoint(serviceUrl) },
    { id: 'anthropic', label: tr('Anthropic 兼容端点', 'Anthropic-compatible endpoint'), url: anthropicEndpoint(serviceUrl, anthropicPort) },
  ];
  async function copyEndpoint(id: string, endpoint: string) {
    try {
      await navigator.clipboard.writeText(endpoint);
      setEndpointCopied(id);
      toast.success(tr('服务地址已复制到剪贴板', 'Endpoint URL copied to clipboard'));
      if (timer.current) clearTimeout(timer.current);
      timer.current = setTimeout(() => setEndpointCopied(null), 1600);
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
  const prefixCacheQueries = Number(runtime?.prefix_cache_queries || 0);
  const prefixCacheHits = Number(runtime?.prefix_cache_hits || 0);
  const prefixCacheHitRate =
    prefixCacheQueries > 0 ? (prefixCacheHits / prefixCacheQueries) * 100 : 0;
  const prefixCacheHotOnly = runtime?.prefix_cache_mode === 'single_device_hot_prefix';
  const prefixCachePersistent = typeof runtime?.prefix_cache_max_bytes === 'number';
  const prefixCacheSupported = runtime?.prefix_cache_supported !== undefined
    ? Number(runtime.prefix_cache_supported) > 0
    : prefixCachePersistent || prefixCacheHotOnly;
  const prefixCacheUnavailableReason = Number(runtime?.prefix_cache_disabled_reason) === 2
    ? tr(
        '连续批处理模式暂不支持 Session KV 缓存',
        'Session KV cache is unavailable with continuous batching',
      )
    : tr(
        '当前模型不支持 Session KV 缓存',
        'The current model does not support Session KV cache',
      );

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
                    <span className="model-identity-trailing">
                      <ModelVendorMark name={name} architecture={name === runtime?.model ? runtime.model_capabilities?.architecture_family || runtime.model_type : undefined} />
                      <span className="runtime-status-pill ready">
                        <i />
                        {selected ? tr('当前', 'Current') : stateLabel}
                      </span>
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
          value={formatNumber(lastPrefill.tokensPerSecond, 1)}
          unit="tok/s"
          detail={`${formatNumber(lastPrefill.milliseconds, 1)} ms · ${tr('输入处理', 'Prompt processing')}`}
          icon="text-forward"
        />
        <MetricTile
          label={tr('解码', 'Decode')}
          value={formatNumber(last?.decode_tps, 1)}
          unit="tok/s"
          detail={tr('输出生成', 'Token generation')}
          icon="waveform"
        />
        <MetricTile
          label={tr('首字延迟', 'TTFT')}
          value={formatNumber(lastTtftMs, 1)}
          unit="ms"
          detail={tr('首次输出耗时', 'Time to first token')}
          icon="clock"
        />
        <MetricTile
          label={tr('前缀复用', 'Prefix reuse')}
          value={
            !prefixCacheSupported
              ? tr('不支持', 'Unavailable')
              : prefixCacheQueries > 0
                ? `${formatNumber(prefixCacheHitRate, 1)}%`
                : '--'
          }
          detail={
            !prefixCacheSupported
              ? prefixCacheUnavailableReason
              : tr(
                  `已恢复 ${formatNumber(runtime?.prefix_cache_hit_tokens || 0)} tokens`,
                  `${formatNumber(runtime?.prefix_cache_hit_tokens || 0)} tokens restored`,
                )
          }
          icon="reuse"
        />
        <MetricTile
          label={tr('内存', 'Memory')}
          value={formatBytes(runtimeMemory)}
          detail={tr('当前模型活动内存', 'Current model active memory')}
          icon="memory"
        />
      </div>
      <SectionLabel title={tr('资源概览', 'Resource overview')} />
      {runtime ? (
        <MemoryHierarchy instances={instances} connectionRevision={connectionRevision}
          memoryCapacityBytes={runtime.runtime_memory_effective_budget_bytes ?? runtime.runtime_memory_budget_bytes} />
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
            <h2>{tr('API 端点', 'API endpoints')}</h2>
          </div>
          <div className="overview-api-list">
            {endpoints.map(({ id, label, url }) => (
              <div className="overview-api-entry" key={id} data-protocol={id}>
                  <strong title={label}>{id === 'openai' ? 'OpenAI' : 'Anthropic'}</strong>
                  {url ? <code title={url}>{url}</code> : <span>{tr('暂不可用', 'Unavailable')}</span>}
                  <button aria-label={tr(`复制 ${id === 'openai' ? 'OpenAI' : 'Anthropic'} 端点`, `Copy ${id === 'openai' ? 'OpenAI' : 'Anthropic'} endpoint`)}
                    className={endpointCopied === id ? 'copied' : ''} disabled={!url}
                    onClick={() => { if (url) void copyEndpoint(id, url); }} type="button">
                    <Icon name={endpointCopied === id ? 'check' : 'copy'} size={14} />
                  </button>
              </div>
            ))}
          </div>
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
