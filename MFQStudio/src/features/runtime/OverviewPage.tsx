/** Provide OverviewPage interface behavior. */
import { localized } from '../../i18n/messages';
import { useTranslation } from 'react-i18next';
import { useEffect, useRef, useState } from 'react';
import { useRuntime } from '../../app/RuntimeProvider';
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
import { openAIEndpoint } from './endpoint';
import { getApiBaseUrl } from '../../shared/api/client';
import { toast } from '../../stores/toastStore';
import { ModelVendorMark } from '../../app/ModelVendorMark';

/** Summarize model availability, runtime memory, request metrics, and endpoint access. */
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
  const { t } = useTranslation();
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
  const endpoint = openAIEndpoint(studio?.service_url || getApiBaseUrl());
  async function copyEndpoint() {
    try {
      await navigator.clipboard.writeText(endpoint);
      setEndpointCopied(true);
      toast.success(localized('runtime:overviewPage.endpointUrlCopiedToClipboard'));
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
    ? t('runtime:overviewPage.sessionKvCacheIsUnavailableWithContinuousBatching')
    : t('runtime:overviewPage.theCurrentModelDoesNotSupportSessionKvCache');

  return (
    <section className="dashboard-view" id="dashboard-overview">
      <ScreenHeader
        title={t('runtime:overviewPage.overview')}
        subtitle={t('runtime:overviewPage.localInferenceStatusAndPerformance')}
        trailing={
          <button disabled={busy} onClick={() => void refreshRuntime()} type="button">
            <Icon name="refresh" size={14} />
            {t('common:refresh')}
          </button>
        }
      />
      <RuntimeHero />
      {availableModelNames.length > 1 && (
        <>
          <SectionLabel
            title={t('runtime:overviewPage.loadedModels')}
            subtitle={t('runtime:overviewPage.availableForInference', { count: availableModelNames.length })}
          />
          <TMPanel className="overview-models-panel">
            <div className="overview-model-grid">
              {availableModelNames.map((name) => {
                const instance = instances.find(
                  (candidate) => candidate.model === name && candidate.state !== 'failed',
                );
                const selected = name === model;
                const stateLabel =
                  instance?.state === 'busy' ? t('runtime:overviewPage.busy') : t('runtime:overviewPage.ready');
                const details = [
                  instance?.devices.join(' + '),
                  instance?.context_size ? `${formatNumber(instance.context_size)} ctx` : null,
                  instance
                    ? t('runtime:overviewPage.sessions', { count: instance.active_sessions })
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
                        {selected ? t('runtime:overviewPage.current') : stateLabel}
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
        title={t('runtime:overviewPage.livePerformance')}
        subtitle={t('runtime:overviewPage.latestRequestThroughputCumulativeCacheReuse')}
      />
      <div className="metric-grid">
        <MetricTile
          label={t('runtime:overviewPage.prefill')}
          value={formatNumber(lastPrefill.tokensPerSecond, 1)}
          unit="tok/s"
          detail={`${formatNumber(lastPrefill.milliseconds, 1)} ms · ${t('runtime:overviewPage.promptProcessing')}`}
          icon="text-forward"
        />
        <MetricTile
          label={t('runtime:overviewPage.decode')}
          value={formatNumber(last?.decode_tps, 1)}
          unit="tok/s"
          detail={t('runtime:overviewPage.tokenGeneration')}
          icon="waveform"
        />
        <MetricTile
          label={t('runtime:overviewPage.ttft')}
          value={formatNumber(lastTtftMs, 1)}
          unit="ms"
          detail={t('runtime:overviewPage.timeToFirstToken')}
          icon="clock"
        />
        <MetricTile
          label={t('runtime:overviewPage.prefixReuse')}
          value={
            !prefixCacheSupported
              ? t('runtime:overviewPage.unavailable')
              : prefixCacheQueries > 0
                ? `${formatNumber(prefixCacheHitRate, 1)}%`
                : '--'
          }
          detail={
            !prefixCacheSupported
              ? prefixCacheUnavailableReason
              : t('runtime:overviewPage.tokensRestored', { tokens: formatNumber(runtime?.prefix_cache_hit_tokens || 0) })
          }
          icon="reuse"
        />
        <MetricTile
          label={t('runtime:overviewPage.memory')}
          value={formatBytes(runtimeMemory)}
          detail={t('runtime:overviewPage.currentModelActiveMemory')}
          icon="memory"
        />
      </div>
      <SectionLabel title={t('runtime:overviewPage.resourceOverview')} />
      {runtime ? (
        <MemoryHierarchy instances={instances} connectionRevision={connectionRevision}
          memoryCapacityBytes={runtime.runtime_memory_effective_budget_bytes ?? runtime.runtime_memory_budget_bytes} />
      ) : (
        <EmptyPanel
          icon="memory"
          title={t('runtime:overviewPage.runtimeMemoryIsUnavailable')}
          message={t('runtime:overviewPage.memoryResidencyAndCacheStateAppearWhenTheServerIsReady')}
        />
      )}
      <div className="overview-footer-grid">
        <TMPanel className="overview-endpoint-panel">
          <div className="overview-panel-title">
            <Icon name="link" size={15} />
            <h2>{t('runtime:overviewPage.openaiCompatibleEndpoint')}</h2>
            <button
              aria-label={endpointCopied ? t('runtime:overviewPage.copied') : t('runtime:overviewPage.copyEndpoint')}
              className={endpointCopied ? 'copied' : ''}
              onClick={() => void copyEndpoint()}
              title={endpointCopied ? t('runtime:overviewPage.copied') : t('runtime:overviewPage.copyEndpoint')}
              type="button"
            >
              <Icon name={endpointCopied ? 'check' : 'copy'} size={14} />
            </button>
          </div>
          <code>{endpoint}</code>
          <p>
            {t('runtime:overviewPage.useThisBaseUrlWithOpenaiSdksTheDefaultLoopbackAddressSends')}
          </p>
        </TMPanel>
        <TMPanel className="overview-session-panel">
          <div className="overview-panel-title">
            <Icon name="chart" size={15} />
            <h2>{t('runtime:overviewPage.session')}</h2>
            <span>{formatDuration(runtime?.uptime_seconds)}</span>
          </div>
          <div className="overview-session-stats">
            <div>
              <strong>{formatNumber(runtime?.total_requests || 0)}</strong>
              <small>{t('runtime:overviewPage.completed')}</small>
            </div>
            <div>
              <strong>{formatNumber(runtime?.total_prompt_tokens || 0)}</strong>
              <small>{t('runtime:overviewPage.prompt')}</small>
            </div>
            <div>
              <strong>{formatNumber(runtime?.total_completion_tokens || 0)}</strong>
              <small>{t('runtime:overviewPage.generated')}</small>
            </div>
          </div>
          <p>
            {formatNumber(runtime?.failed_requests || 0)} {t('runtime:overviewPage.failedRequests')} ·{' '}
            {formatNumber(runtime?.active_requests || 0)} {t('runtime:overviewPage.active')}
          </p>
        </TMPanel>
      </div>
    </section>
  );
}
