import { useRef, useState } from 'react';
import { PauseIcon, PlayIcon } from '@phosphor-icons/react';
import { jobsApi } from '../../shared/api/resources/jobs';
import type { JobResource } from '../../shared/api/types';
import { useRuntime } from '../../app/RuntimeProvider';
import { useConnectionScope } from '../../app/useConnectionScope';
import { useSettings } from '../settings/SettingsProvider';
import { Icon, ScreenHeader, TMPanel } from '../../app/display';
import { errorMessage, formatNumber } from '../../app/formatters';
import { isTerminalJob } from '../jobs/jobSchema';
import { studioConfirm } from '../../studio';
import { toast } from '../../stores/toastStore';
import { useJobStore } from '../../stores/jobStore';
import { useLogHistory } from './useLogHistory';
import { JobDetails, LogDetails, RequestDetails } from './LogDetails';

type LogTab = 'events' | 'requests' | 'jobs';

function jobModelName(job: JobResource): string | undefined {
  const resolved = job.result?.model;
  return typeof resolved === 'string' && resolved ? resolved
    : typeof job.payload.model === 'string' ? job.payload.model : undefined;
}

export function LogsPage() {
  const { instances, connectionRevision } = useRuntime();
  return <LogConsole key={connectionRevision} instances={instances || []} />;
}

function LogConsole({ instances }: { instances: ReturnType<typeof useRuntime>['instances'] }) {
  const { refreshRuntime, refreshError } = useRuntime();
  const connectionScope = useConnectionScope();
  const { tr } = useSettings();
  const [tab, setTab] = useState<LogTab>('events');
  const [paused, setPaused] = useState(false);
  const [query, setQuery] = useState('');
  const [source, setSource] = useState('');
  const [filter, setFilter] = useState('');
  const [range, setRange] = useState('24');
  const [selected, setSelected] = useState('');
  const [cleanupBusy, setCleanupBusy] = useState(false);
  const cleanupPending = useRef(false);
  const [refreshBusy, setRefreshBusy] = useState(false);
  const [manualError, setManualError] = useState('');
  const [manualRefreshFailed, setManualRefreshFailed] = useState(false);
  const refreshPending = useRef(false);
  const history = useLogHistory(paused);
  const error = [history.error, manualError || (manualRefreshFailed ? refreshError : '')].filter(Boolean).join(' · ');
  const sourceNames = new Map(instances.map((item) => [item.id, item.model]));
  history.requests.forEach((item) => { if (item.instance_id && item.model) sourceNames.set(item.instance_id, item.model); });
  history.logs.forEach((item) => {
    if (item.instance_id && typeof item.fields.model === 'string' && item.fields.model && !sourceNames.has(item.instance_id))
      sourceNames.set(item.instance_id, item.fields.model);
  });
  history.jobs.forEach((job) => {
    const model = jobModelName(job);
    if (!model) return;
    for (const id of [job.payload.instance_id, job.result?.instance_id]) {
      if (typeof id === 'string' && id && !sourceNames.has(id)) sourceNames.set(id, model);
    }
  });
  [...history.logs.map((item) => item.instance_id),
    ...history.jobs.flatMap((job) => [job.payload.instance_id, job.result?.instance_id])].forEach((id) => {
    if (typeof id === 'string' && id && !sourceNames.has(id)) sourceNames.set(id,
      `${tr('历史实例', 'Past instance')} · ${id.slice(0, 8)}`);
  });
  const search = query.trim().toLocaleLowerCase();
  const cutoff = range ? (paused ? history.updatedAt ?? Date.now() : Date.now()) - Number(range) * 3600000 : 0;
  const inRange = (time: string | number) => (typeof time === 'number' ? time * 1000 : Date.parse(time)) >= cutoff;
  const matches = (value: unknown) => !search || JSON.stringify(value).toLocaleLowerCase().includes(search);
  const events = history.logs.filter((item) => (!source || item.instance_id === source) && (!filter || item.level === filter)
    && inRange(item.created_at) && matches([item.message, item.fields, sourceNames.get(item.instance_id || ''), item.instance_id]));
  const requests = history.requests.filter((item) => (!source || item.instance_id === source) && (!filter || item.request.finish_reason === filter)
    && inRange(item.request.completed_at ?? item.captured_at) && matches(item));
  const jobs = history.jobs.filter((item) => {
    const targets = [item.payload.instance_id, item.result?.instance_id].filter((id): id is string => typeof id === 'string' && !!id);
    return (!source || (targets.length ? targets.includes(source) : jobModelName(item) === sourceNames.get(source)))
      && (!filter || (filter === 'active' ? !isTerminalJob(item) : item.status === filter)) && inRange(item.updated_at) && matches(item);
  })
    .sort((a, b) => Date.parse(b.updated_at) - Date.parse(a.updated_at));
  const visible = tab === 'events' ? events : tab === 'requests' ? requests : jobs;
  const total = tab === 'events' ? history.logs.length : tab === 'requests' ? history.requests.length : history.jobs.length;
  const statusLabel = (status: string) => ({ queued: tr('排队中', 'Queued'), running: tr('运行中', 'Running'), cancelling: tr('取消中', 'Cancelling'),
    succeeded: tr('已完成', 'Completed'), failed: tr('失败', 'Failed'), interrupted: tr('已中断', 'Interrupted'), cancelled: tr('已取消', 'Cancelled') })[status] || status;
  const jobLabel = (kind: string) => ({ 'model.load': tr('加载模型', 'Load model'), 'model.unload': tr('卸载模型', 'Unload model'),
    'hub.download': tr('下载模型', 'Download model'), 'model.download': tr('下载模型', 'Download model'),
    'download.huggingface': tr('下载模型', 'Download model'), 'download.modelscope': tr('下载模型', 'Download model'),
    'dataset.download': tr('下载官方测评集合', 'Download official test collection'),
    'evaluate.wikitext2': 'WT2 KLD / Top1', 'evaluate.accuracy': tr('任务测评', 'Task evaluation'),
    'benchmark.inference': tr('推理测速', 'Inference benchmark'), 'benchmark.kernel': tr('算子测速', 'Kernel benchmark'),
    'runtime.memory.configure': tr('调整内存预算', 'Adjust memory budget') })[kind] || kind;
  const time = (value: string | number) => new Date(typeof value === 'number' ? value * 1000 : value)
    .toLocaleString(undefined, { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
  const metric = (value: number | undefined, suffix = '') => value == null ? '--' : `${formatNumber(value, 1)}${suffix}`;
  function selectTab(next: LogTab) { setTab(next); setFilter(''); setSelected(''); }
  function toggle(id: string) { setSelected((current) => current === id ? '' : id); }
  async function refresh() {
    if (refreshPending.current) return;
    const current = connectionScope();
    refreshPending.current = true;
    setRefreshBusy(true); setManualError(''); setManualRefreshFailed(false);
    try {
      const refreshed = await refreshRuntime();
      if (!current()) return;
      setManualRefreshFailed(!refreshed);
      history.refresh();
    } catch (cause) {
      if (current()) { setManualError(errorMessage(cause)); history.refresh(); }
    } finally {
      refreshPending.current = false;
      if (current()) setRefreshBusy(false);
    }
  }
  async function cleanup(id?: string) {
    const current = connectionScope();
    if (cleanupPending.current) return;
    cleanupPending.current = true;
    setCleanupBusy(true);
    try {
      if (!(await studioConfirm(id
        ? tr('删除这条任务记录？不会删除模型文件。', 'Delete this job record? Model files will be kept.')
        : tr('清理所有已结束的任务记录？不会删除模型文件。', 'Clear all finished job records? Model files will be kept.'))) || !current()) return;
      if (id) await jobsApi.deleteJob(id); else await jobsApi.clearCompletedJobs();
      if (!current()) return;
      const kept = useJobStore.getState().jobs.filter((job) => id ? job.id !== id : !isTerminalJob(job));
      useJobStore.getState().setJobs(kept);
      setSelected('');
      await refreshRuntime();
      if (!current()) return;
      history.refresh();
      toast.success(tr('任务记录已清理', 'Job records cleaned up'));
    } catch (cause) { if (current()) toast.error(errorMessage(cause)); }
    finally { cleanupPending.current = false; if (current()) setCleanupBusy(false); }
  }
  function exportRecords() {
    try {
      const url = URL.createObjectURL(new Blob([JSON.stringify(visible, null, 2)], { type: 'application/json' }));
      const anchor = document.createElement('a');
      anchor.href = url; anchor.download = `mfq-${tab}-${new Date().toISOString().slice(0, 10)}.json`;
      anchor.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (cause) { toast.error(errorMessage(cause)); }
  }
  const filterOptions = tab === 'events'
    ? [['error', tr('错误', 'Error')], ['warning', tr('警告', 'Warning')], ['info', tr('信息', 'Info')], ['debug', tr('调试', 'Debug')]]
    : tab === 'jobs' ? ['active', 'succeeded', 'failed', 'interrupted', 'cancelled'].map((value) => [value, value === 'active' ? tr('进行中', 'Active') : statusLabel(value)])
      : [...new Set(history.requests.map((item) => item.request.finish_reason).filter(Boolean))].map((value) => [value!, value!]);
  return <section className="dashboard-view logs-view">
    <ScreenHeader title={tr('日志', 'Logs')} subtitle={tr('查看服务事件、请求性能与任务记录。', 'Inspect service events, request performance, and job records.')}
      trailing={<>
        <button type="button" aria-pressed={paused} onClick={() => setPaused((current) => !current)} disabled={!history.ready}>
          {paused ? <PlayIcon size={13} /> : <PauseIcon size={13} />}{paused ? tr('继续更新', 'Resume updates') : tr('暂停更新', 'Pause updates')}</button>
        <button type="button" onClick={() => void refresh()} disabled={refreshBusy || history.loading || !history.ready}><Icon name="refresh" size={13} />{tr('刷新', 'Refresh')}</button>
      </>} />
    <TMPanel className="logs-console">
      <div className="logs-console-heading">
        <div className="logs-tabs" role="tablist" aria-label={tr('日志类别', 'Log categories')}>
          {([['events', tr('服务事件', 'Service events'), history.logs.length], ['requests', tr('请求记录', 'Requests'), history.requests.length],
            ['jobs', tr('任务记录', 'Jobs'), history.jobs.length]] as const).map(([id, label, count]) =>
            <button key={id} id={`logs-tab-${id}`} role="tab" type="button" aria-selected={tab === id} tabIndex={tab === id ? 0 : -1}
              aria-controls="logs-records" onClick={() => selectTab(id)} onKeyDown={(event) => {
                const tabs: LogTab[] = ['events', 'requests', 'jobs'];
                const next = event.key === 'ArrowRight' ? tabs[(tabs.indexOf(id) + 1) % 3]
                  : event.key === 'ArrowLeft' ? tabs[(tabs.indexOf(id) + 2) % 3]
                    : event.key === 'Home' ? tabs[0] : event.key === 'End' ? tabs[2] : null;
                if (next) { event.preventDefault(); selectTab(next); document.getElementById(`logs-tab-${next}`)?.focus(); }
              }}>{label}<span>{count}</span></button>)}
        </div>
        <span className={`logs-live-state${paused || !history.ready || error ? ' paused' : ''}`}><i />
          {!history.ready ? tr('未连接', 'Disconnected') : error ? tr('刷新失败', 'Refresh failed') : paused ? tr('已暂停', 'Paused') : tr('实时更新', 'Live updates')}</span>
      </div>
      <div className="logs-filters">
        <label className="logs-search"><span>{tr('搜索', 'Search')}</span><input type="search" value={query} onChange={(event) => setQuery(event.target.value)}
          placeholder={tr('搜索消息、模型或 ID', 'Search messages, models, or IDs')} /></label>
        <label><span>{tr('来源', 'Source')}</span><select value={source} onChange={(event) => setSource(event.target.value)}>
          <option value="">{tr('全部模型与服务', 'All models and service')}</option>
          {[...sourceNames].map(([id, name]) => <option value={id} key={id}>{name} · {id.slice(0, 8)}</option>)}</select></label>
        <label><span>{tab === 'events' ? tr('级别', 'Level') : tab === 'jobs' ? tr('状态', 'Status') : tr('停止原因', 'Finish reason')}</span>
          <select value={filter} onChange={(event) => setFilter(event.target.value)}><option value="">{tr('全部', 'All')}</option>
            {filterOptions.map(([id, label]) => <option value={id} key={id}>{label}</option>)}</select></label>
        <label><span>{tr('时间', 'Time')}</span><select value={range} onChange={(event) => setRange(event.target.value)}>
          <option value="1">{tr('最近 1 小时', 'Last hour')}</option><option value="24">{tr('最近 24 小时', 'Last 24 hours')}</option>
          <option value="168">{tr('最近 7 天', 'Last 7 days')}</option><option value="">{tr('全部时间', 'All time')}</option></select></label>
      </div>
      <div className="logs-list-heading"><span>{tr(`${visible.length} / ${total} 条 · 最新在前`, `${visible.length} / ${total} records · newest first`)}</span>
        <div>{(query || source || filter || range !== '24') && <button type="button" onClick={() => { setQuery(''); setSource(''); setFilter(''); setRange('24'); }}>
          {tr('重置筛选', 'Reset filters')}</button>}
          {tab === 'jobs' && <button type="button" className="danger" disabled={cleanupBusy || !history.jobs.some(isTerminalJob)} onClick={() => void cleanup()}>
            {tr('清理已结束记录', 'Clear finished records')}</button>}
          <button type="button" disabled={!visible.length} onClick={exportRecords}><Icon name="download" size={13} />{tr('导出', 'Export')}</button></div>
      </div>
      {error && <p className="logs-error" role="status">{error}{tr(' · 保留上次成功读取的记录', ' · Showing previously retrieved records')}</p>}
      <div id="logs-records" className="logs-records" role="tabpanel" aria-labelledby={`logs-tab-${tab}`}>
        {visible.length === 0 && <div className="logs-empty"><Icon name="scroll" size={28} /><strong>
          {history.loading ? tr('正在读取记录…', 'Loading records…') : !history.ready ? tr('服务未连接', 'Service disconnected') : tr('没有匹配的记录', 'No matching records')}</strong>
          <span>{tr('可以调整筛选条件，或等待新的服务活动。', 'Adjust the filters or wait for new service activity.')}</span></div>}
        {tab === 'events' && events.map((entry) => <div className={`log-record log-event-record ${entry.level}`} key={entry.sequence}>
          <button type="button" className="log-record-toggle" aria-expanded={selected === String(entry.sequence)} onClick={() => toggle(String(entry.sequence))}>
            <time>{time(entry.created_at)}</time><span className={`log-level ${entry.level}`}>{entry.level.toUpperCase()}</span>
            <span className="log-record-copy"><strong>{entry.message}</strong><small>{sourceNames.get(entry.instance_id || '') || tr('服务', 'Service')}
              {typeof entry.fields.source === 'string' && ` · ${entry.fields.source}`}</small></span><span className="log-record-chevron">›</span></button>
          {selected === String(entry.sequence) && <LogDetails value={entry} />}
        </div>)}
        {tab === 'requests' && requests.map((record) => <div className="log-record log-request-record" key={record.key}>
          <button type="button" className="log-record-toggle" aria-expanded={selected === record.key} onClick={() => toggle(record.key)}>
            <time>{time(record.request.completed_at ?? record.captured_at)}</time><span className="log-record-copy"><strong>{record.model || record.request.endpoint || tr('推理请求', 'Inference request')}</strong>
              <small>{record.request.id}</small></span>
            <span className="log-request-metric"><b>{metric(record.request.decode_tps, ' tok/s')}</b><small>{metric(record.request.prompt_tokens)} → {metric(record.request.completion_tokens)}</small></span>
            <span className="log-record-chevron">›</span></button>
          {selected === record.key && <RequestDetails record={record} />}
        </div>)}
        {tab === 'jobs' && jobs.map((job) => <div className="log-record log-job-record" key={job.id}>
          <button type="button" className="log-record-toggle" aria-expanded={selected === job.id} onClick={() => toggle(job.id)}>
            <time>{time(job.updated_at)}</time><span className={`log-job-state ${job.status}`}>{statusLabel(job.status)}</span>
            <span className="log-record-copy"><strong>{jobLabel(job.kind)}</strong><small>{String(jobModelName(job)
              || (typeof job.payload.instance_id === 'string' && sourceNames.get(job.payload.instance_id))
              || job.payload.repo_id || job.payload.repository || job.payload.dataset || job.id)}</small></span>
            <span className="log-job-progress"><progress max={1} value={Math.max(0, Math.min(1, job.progress))} />{formatNumber(job.progress * 100)}%</span>
            <span className="log-record-chevron">›</span></button>
          {selected === job.id && <JobDetails job={job} onRemove={() => void cleanup(job.id)} busy={cleanupBusy} refreshAt={history.updatedAt} />}
        </div>)}
      </div>
      <footer className="logs-footer"><span>{tr('每 4 秒刷新 · 服务事件 / 请求快照各保留最近 500 条', 'Refreshes every 4 seconds · latest 500 service events / request snapshots')}</span>
        <span>{history.updatedAt ? `${tr('最近读取', 'Last retrieved')} ${new Date(history.updatedAt).toLocaleTimeString()}` : '--'}</span></footer>
    </TMPanel>
  </section>;
}
