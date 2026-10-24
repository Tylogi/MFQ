import { useEffect, useMemo, useRef, useState } from 'react';
import { useRuntime } from '../../app/RuntimeProvider';
import { useSettings } from '../settings/SettingsProvider';
import {
  Icon,
  ScreenHeader,
  SectionLabel,
  TMPanel,
  ModelMonogram,
  EmptyPanel,
} from '../../app/display';
import { errorMessage, formatNumber } from '../../app/formatters';
import { runtimeModelNames } from './modelSelection';
import { ServingStatisticsPanel, ServingTotalsPanel } from './ServingStatisticsPanel';
import { useServingStatistics } from './useServingStatistics';
import { RuntimeHero } from './RuntimeHero';
import { MemoryHierarchy } from './MemoryHierarchy';
import { residentMemoryCapacity } from './memoryArchitecture';
import { anthropicEndpoint, openAIEndpoint } from './endpoint';
import { getApiBaseUrl, getApiToken } from '../../shared/api/client';
import { runtimeApi } from '../../shared/api/resources/runtime';
import type { RuntimeInstance, RuntimeStatus } from '../../shared/api/types';
import { toast } from '../../stores/toastStore';
import { ModelVendorMark } from '../../app/ModelVendorMark';
import { RuntimeConcurrencyPanel } from './RuntimeConcurrencyPanel';

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
  const [scope, setScope] = useState<'session' | 'alltime'>('session');
  const { data: servingStatistics, error: statisticsError } = useServingStatistics(scope, connectionRevision);
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
  const availableModelNames = useMemo(
    () => runtimeModelNames(models, instances),
    [models, instances],
  );
  // The serving poll re-renders this page twice a second. Keeping the cards as
  // stable elements lets React skip rebuilding the grid when only the
  // statistics changed.
  const modelCards = useMemo(
    () =>
      availableModelNames.map((name) => {
        const instance = instances.find(
          (candidate) => candidate.model === name && candidate.state !== 'failed',
        );
        const selected = name === model;
        const stateLabel = instance?.state === 'busy' ? tr('使用中', 'Busy') : tr('就绪', 'Ready');
        const details = [
          instance?.devices.join(' + '),
          instance?.context_size ? `${formatNumber(instance.context_size)} ctx` : null,
          instance
            ? tr(`${instance.active_sessions} 个会话`, `${instance.active_sessions} sessions`)
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
      }),
    [availableModelNames, instances, model, busy, runtime, selectModel, tr],
  );
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
            <div className="overview-model-grid">{modelCards}</div>
          </TMPanel>
        </>
      )}
      <ServingStatisticsPanel data={servingStatistics} error={statisticsError} scope={scope} onScopeChange={setScope} />
      <RuntimeConcurrencyPanel />
      <SectionLabel title={tr('资源概览', 'Resource overview')} />
      {runtime ? (
        <MemoryHierarchy instances={instances} runtime={runtime} connectionRevision={connectionRevision}
          memoryCapacityBytes={residentMemoryCapacity(runtime)} />
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
        <ServingTotalsPanel data={servingStatistics} scope={scope} runtime={runtime} />
      </div>
    </section>
  );
}
