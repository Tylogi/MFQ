import { Icon, MetricTile, SectionLabel, TMPanel } from '../../app/display';
import { formatDuration, formatNumber } from '../../app/formatters';
import type { RuntimeStatus, ServingStatistics } from '../../shared/api/contracts/runtime';
import { useSettings } from '../settings/SettingsProvider';
import './serving.css';

type StatisticsProps = { data: ServingStatistics | null; scope: ServingStatistics['scope'] };

export function ServingStatisticsPanel({ data, error, scope, onScopeChange }: StatisticsProps & {
  error: string | null;
  onScopeChange: (scope: ServingStatistics['scope']) => void;
}) {
  const { tr } = useSettings();
  const totals = data?.totals;
  const active = data?.activity.filter(item => item.active_requests > 0 || item.queued_requests > 0) ?? [];
  const concurrency = data?.activity.reduce((total, item) => total + item.active_requests, 0);
  const queued = data?.activity.reduce((total, item) => total + item.queued_requests, 0);
  return <>
    <div className="serving-heading">
      <SectionLabel title={tr('服务统计', 'Serving statistics')} subtitle={tr('已完成请求 · 所有模型', 'Completed requests · all models')} />
      <div className="serving-scope" role="group" aria-label={tr('统计范围', 'Statistics scope')}>
        <button type="button" aria-pressed={scope === 'session'} onClick={() => onScopeChange('session')}>{tr('本次运行', 'Session')}</button>
        <button type="button" aria-pressed={scope === 'alltime'} onClick={() => onScopeChange('alltime')}>{tr('全部历史', 'All time')}</button>
      </div>
    </div>
    {error && <p className="serving-error" role="status">{tr('统计暂未更新', 'Statistics could not be refreshed')} · {error}</p>}
    <div className="metric-grid">
      <MetricTile label={tr('预填充', 'Prefill')} value={formatNumber(totals?.prefill_tps ?? undefined, 1)} unit="tok/s"
        detail={tr('输入处理', 'Prompt processing')} icon="text-forward" />
      <MetricTile label={tr('解码', 'Decode')} value={formatNumber(totals?.decode_tps ?? undefined, 1)} unit="tok/s"
        detail={tr('输出生成', 'Token generation')} icon="waveform" />
      <MetricTile label={tr('首字延迟', 'TTFT')} value={formatNumber(totals?.ttft_ms ?? undefined, 1)} unit="ms"
        detail={tr('已完成请求平均值', 'Completed request average')} icon="clock" />
      <MetricTile label={tr('前缀复用', 'Prefix reuse')} value={totals?.cache_hit_rate == null ? '--' : `${formatNumber(totals.cache_hit_rate, 1)}%`}
        detail={totals?.completed_requests ? tr(`已复用 ${formatNumber(totals.cached_tokens)} tokens`, `${formatNumber(totals.cached_tokens)} tokens reused`) : '--'} icon="reuse" />
      <MetricTile label={tr('并发数', 'Concurrency')} value={formatNumber(concurrency)}
        detail={queued ? tr(`${formatNumber(queued)} 个排队`, `${formatNumber(queued)} queued`) : tr('当前活动请求', 'Active requests')} icon="queue" />
    </div>
    {active.length > 0 && <>
      <SectionLabel title={tr('当前请求', 'Current requests')} />
      <TMPanel className="serving-activity-panel">{active.map(item => <div className="serving-model-activity" key={item.instance_id}>
        <div className="serving-model-heading"><strong title={item.model}>{item.model}</strong>
          <span>{tr(`${item.active_requests} 个活动请求`, `${item.active_requests} active`)}</span>
          {item.queued_requests > 0 && <span>{tr(`${item.queued_requests} 个排队`, `${item.queued_requests} queued`)}</span>}
        </div>
        {item.requests.length === 0 && <div className="serving-request"><span>{tr('处理中', 'Processing')}</span></div>}
        {item.requests.map(request => <div className="serving-request" key={request.id}>
          <code title={request.id}>{request.id.slice(-12)}</code>
          <span className="serving-phase">{request.phase === 'generating' ? tr('生成中', 'Generating') : tr('预填充中', 'Prefilling')}</span>
          <span>{formatNumber(request.phase === 'generating' ? request.generated_tokens : request.processed_tokens)} tokens</span>
          <span>{formatNumber(request.phase === 'generating' ? request.decode_tps : request.prefill_tps, 1)} tok/s</span>
          <span>{formatDuration(request.elapsed_seconds)}</span>
          {(item.sample_age_seconds ?? 0) > 5 && <small>{tr('状态延迟', 'Status delayed')}</small>}
        </div>)}
      </div>)}</TMPanel>
    </>}
  </>;
}

export function ServingTotalsPanel({ data, scope, runtime }: StatisticsProps & { runtime: RuntimeStatus | null }) {
  const { tr } = useSettings();
  const uptime = data ? Math.max(0, Date.now() / 1000 - data.started_at) : runtime?.uptime_seconds;
  const active = data?.activity.reduce((total, item) => total + item.active_requests, 0) ?? runtime?.active_requests;
  return <TMPanel className="overview-session-panel">
    <div className="overview-panel-title">
      <Icon name="chart" size={15} />
      <h2>{scope === 'session' ? tr('本次运行', 'Session') : tr('全部历史', 'All time')}</h2>
      <span title={tr('本次运行时长', 'Session uptime')}>{formatDuration(uptime)}</span>
    </div>
    <div className="overview-session-stats">
      <div><strong>{formatNumber(data?.totals.completed_requests)}</strong><small>{tr('已完成', 'Completed')}</small></div>
      <div><strong>{formatNumber(data?.totals.prompt_tokens)}</strong><small>{tr('输入', 'Input')}</small></div>
      <div><strong>{formatNumber(data?.totals.completion_tokens)}</strong><small>{tr('输出', 'Output')}</small></div>
    </div>
    <p>{formatNumber(runtime?.failed_requests)} {tr('个失败请求', 'failed requests')} ·{' '}
      {formatNumber(active)} {tr('个活动请求', 'active')}</p>
  </TMPanel>;
}
