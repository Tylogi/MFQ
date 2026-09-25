/** 运行概况条汇总模型加载任务与实例状态，并提供业务页面导航。 */
import { useMemo } from 'react';
import { useNavigate } from 'react-router';
import { useRuntime } from '../../app/RuntimeProvider';
import { useJobStore } from '../../stores/jobStore';
import { useSettings } from '../settings/SettingsProvider';
import { Icon, TMPanel, ModelMonogram } from '../../app/display';
import { formatNumber, formatDuration } from '../../app/formatters';
import { STUDIO_PATHS } from '../../navigation';

/** 按加载任务、当前实例和失败记录推导概览状态，避免仅凭模型名称误判就绪。 */
export function RuntimeHero() {
  const { runtime, instances, studio } = useRuntime();
  const jobs = useJobStore((state) => state.jobs);
  const { tr } = useSettings();
  const navigate = useNavigate();
  const openStudioPage = () => navigate(STUDIO_PATHS.models);
  const openChatPage = () => navigate(STUDIO_PATHS.chat);
  const runtimeModelName = runtime?.model || 'Empty';
  const modelHero = useMemo(() => {
    const activeLoadJob = jobs.find(
      (job) =>
        job.kind === 'model.load' && ['queued', 'running', 'cancelling'].includes(job.status),
    );
    const latestLifecycleJob = jobs.find(
      (job) => job.kind === 'model.load' || job.kind === 'model.unload',
    );
    const currentInstance = instances.find((instance) => instance.id === runtime?.instance_id);
    const loadingInstance = [...instances]
      .reverse()
      .find((instance) => instance.state === 'loading');
    const readyInstance =
      currentInstance?.state === 'ready' || currentInstance?.state === 'busy'
        ? currentInstance
        : [...instances]
            .reverse()
            .find((instance) => instance.state === 'ready' || instance.state === 'busy');
    const failedInstance =
      currentInstance?.state === 'failed'
        ? currentInstance
        : [...instances].reverse().find((instance) => instance.state === 'failed');
    const jobModel =
      typeof activeLoadJob?.payload.model === 'string' ? activeLoadJob.payload.model : '';
    const runtimeState = String(runtime?.runtime_state || '').toLowerCase();

    if (runtime?.reloading || runtimeState === 'loading' || loadingInstance || activeLoadJob) {
      return {
        name: loadingInstance?.model || jobModel || runtimeModelName,
        state: 'loading' as const,
      };
    }
    if (runtimeState === 'failed' || currentInstance?.state === 'failed') {
      return {
        name: currentInstance?.model || runtimeModelName,
        state: 'failed' as const,
      };
    }
    if (runtime?.model || readyInstance) {
      return {
        name: runtime?.model || readyInstance?.model || runtimeModelName,
        state: 'ready' as const,
      };
    }
    if (failedInstance) {
      return {
        name: failedInstance.model || runtimeModelName,
        state: 'failed' as const,
      };
    }
    if (latestLifecycleJob?.kind === 'model.load' && latestLifecycleJob.status === 'failed') {
      const failedJobModel =
        typeof latestLifecycleJob.payload.model === 'string'
          ? latestLifecycleJob.payload.model
          : '';
      return {
        name: failedJobModel || runtimeModelName,
        state: 'failed' as const,
      };
    }
    return { name: runtimeModelName, state: 'idle' as const };
  }, [instances, jobs, runtime, runtimeModelName]);
  const modelHeroStatus =
    modelHero.state === 'loading'
      ? tr('加载中', 'Loading')
      : modelHero.state === 'ready'
        ? tr('运行中', 'Running')
        : modelHero.state === 'failed'
          ? tr('加载失败', 'Failed')
          : tr('空闲', 'Idle');
  return (
    <TMPanel className="runtime-hero">
      <ModelMonogram name={modelHero.name} state={modelHero.state} />
      <div className="runtime-hero-copy">
        <div>
          <h2>{modelHero.name}</h2>
          <span className={`runtime-status-pill ${modelHero.state}`}>
            <i />
            {modelHeroStatus}
          </span>
        </div>
        <p className="runtime-endpoint">{studio?.service_url || 'http://127.0.0.1:8090'}</p>
        <small>
          {runtime?.model
            ? `${runtime?.model_type || 'MFQ'} · ${formatNumber(runtime?.max_context)} ${tr('上下文', 'context')} · ${formatDuration(runtime?.uptime_seconds)}`
            : tr('加载本地模型后即可开始推理。', 'Load a local model to begin inference.')}
        </small>
      </div>
      <div className="runtime-hero-actions">
        <button onClick={() => openStudioPage()} type="button">
          <Icon name="folder" size={15} />
          {tr('模型', 'Models')}
        </button>
        <button className="primary" onClick={openChatPage} type="button">
          <Icon name="chat" size={15} />
          {tr('对话', 'Chat')}
        </button>
      </div>
    </TMPanel>
  );
}
