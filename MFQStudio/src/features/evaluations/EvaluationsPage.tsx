import { useEffect, useState } from 'react';
import { Icon } from '../../app/display';
import { evaluationsApi } from '../../shared/api/resources/evaluations';
import { modelsApi } from '../../shared/api/resources/models';
import { jobsApi } from '../../shared/api/resources/jobs';
import { useRuntime } from '../../app/RuntimeProvider';
import { useConnectionScope } from '../../app/useConnectionScope';
import { errorMessage } from '../../app/formatters';
import { useSettings } from '../settings/SettingsProvider';
import type { DatasetResource, EvaluationResult, EvaluationComparison, EvaluationTools, ModelArtifact, JobResource, OfficialDataset } from '../../shared/api/types';
import { toast } from '../../stores/toastStore';
import { ModelVendorMark } from '../../app/ModelVendorMark';
import { QualityForm, PerformanceForm, AccuracyForm } from './EvaluationForms';
import { ThroughputTable, AnswerDetails, exportEvaluation, resultObjects } from './EvaluationDetails';
import { OfficialDatasets } from './OfficialDatasets';
import { taskBenchmarks, taskBenchmarkGroups, type TaskBenchmark } from './benchmarkTasks';
import { BenchmarkMark, BenchmarkGroupMark } from './BenchmarkMarks';
import type { GeneratedWt2Reference } from './Wt2ReferenceGenerator';

const activeJob = (job: JobResource) => ['queued', 'running', 'cancelling'].includes(job.status);
const evaluationJob = (job: JobResource) => ['evaluate.wikitext2', 'reference.wikitext2', 'benchmark.inference', 'evaluate.accuracy', 'dataset.download'].includes(job.kind);
const metricKeys = ['kld', 'top1_agreement', 'decode_prefill_tps', 'decode_decode_tps', 'mtp_prefill_tps', 'mtp_decode_tps', 'mtp_acceptance_rate', 'accuracy', 'correct', 'sample_count', 'question_count'] as const;

export function EvaluationsPage() {
  const { tr } = useSettings();
  const { instances, addJob } = useRuntime();
  const connectionScope = useConnectionScope();
  const [tab, setTab] = useState<'quality' | 'performance' | TaskBenchmark['id']>('quality');
  const [view, setView] = useState<'configure' | 'datasets' | 'results'>('configure');
  const [busy, setBusy] = useState(false);
  const [datasets, setDatasets] = useState<DatasetResource[]>([]);
  const [officialCatalog, setOfficialCatalog] = useState<OfficialDataset[]>([]);
  const [evaluations, setEvaluations] = useState<EvaluationResult[]>([]);
  const [models, setModels] = useState<ModelArtifact[]>([]);
  const [tools, setTools] = useState<EvaluationTools | null>(null);
  const [jobs, setJobs] = useState<JobResource[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [comparison, setComparison] = useState<EvaluationComparison | null>(null);
  const [resultKind, setResultKind] = useState('all');
  const [generatedReference, setGeneratedReference] = useState<GeneratedWt2Reference | null>(null);
  const metricLabels: Record<string, string> = {
    kld: 'KLD', top1_agreement: tr('Top1 一致率', 'Top1 agreement'),
    decode_prefill_tps: tr('普通 prefill', 'Ordinary prefill'), decode_decode_tps: tr('普通 decode', 'Ordinary decode'),
    mtp_prefill_tps: 'MTP prefill', mtp_decode_tps: 'MTP decode', mtp_acceptance_rate: tr('MTP 接受率', 'MTP acceptance'),
    accuracy: tr('正确率', 'Accuracy'), correct: tr('正确题数', 'Correct answers'), sample_count: tr('实测题数', 'Questions tested'), question_count: tr('实测题数', 'Questions tested'),
  };
  const statusLabels: Record<string, string> = {
    queued: tr('排队中', 'Queued'), running: tr('运行中', 'Running'), cancelling: tr('取消中', 'Cancelling'),
    succeeded: tr('完成', 'Succeeded'), failed: tr('失败', 'Failed'), cancelled: tr('已取消', 'Cancelled'), interrupted: tr('已中断', 'Interrupted'),
  };
  function metricValue(key: string, value: unknown): string {
    if (typeof value !== 'number' || !Number.isFinite(value)) return '—';
    if (key === 'top1_agreement' || key === 'mtp_acceptance_rate' || key === 'accuracy') return `${(value * 100).toFixed(2)}%`;
    if (key === 'correct' || key === 'sample_count' || key === 'question_count') return String(value);
    return key.endsWith('_tps') ? `${value.toFixed(1)} tok/s` : value.toFixed(6);
  }
  async function refreshResults() {
    const current = connectionScope();
    try {
      const [nextJobs, nextEvaluations, nextDatasets, nextTools] = await Promise.all([jobsApi.jobs(100), evaluationsApi.evaluations(), evaluationsApi.datasets(), evaluationsApi.tools()]);
      if (!current()) return;
      setJobs(nextJobs.filter(evaluationJob)); setEvaluations(nextEvaluations); setDatasets(nextDatasets); setTools(nextTools);
    } catch (cause) { if (current()) toast.error(errorMessage(cause)); }
  }
  useEffect(() => {
    let active = true;
    void Promise.all([evaluationsApi.datasets(), evaluationsApi.evaluations(), modelsApi.modelArtifacts(), evaluationsApi.tools(), jobsApi.jobs(100), evaluationsApi.catalog()])
      .then(([nextDatasets, nextEvaluations, nextModels, nextTools, nextJobs, nextCatalog]) => {
        if (!active) return;
        setDatasets(nextDatasets); setEvaluations(nextEvaluations); setModels(nextModels.filter((item) => item.complete && !item.error && item.format === 'mfq'));
        setTools(nextTools); setJobs(nextJobs.filter(evaluationJob));
        setOfficialCatalog(nextCatalog);
      }).catch((cause) => { if (active) toast.error(errorMessage(cause)); });
    return () => { active = false; };
  }, []);
  const hasActiveJobs = jobs.some(activeJob);
  useEffect(() => {
    if (!hasActiveJobs) return;
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const [nextJobs, nextEvaluations, nextDatasets, nextTools] = await Promise.all([jobsApi.jobs(100), evaluationsApi.evaluations(), evaluationsApi.datasets(), evaluationsApi.tools()]);
        if (!active) return;
        setJobs(nextJobs.filter(evaluationJob)); setEvaluations(nextEvaluations);
        setDatasets(nextDatasets);
        setTools(nextTools);
      } catch (cause) { if (active) toast.error(errorMessage(cause)); }
      if (active) timer = setTimeout(() => void poll(), 1000);
    }
    timer = setTimeout(() => void poll(), 1000);
    return () => { active = false; clearTimeout(timer); };
  }, [hasActiveJobs]);
  async function submit(kind: string, payload: Record<string, unknown>) {
    const current = connectionScope();
    setBusy(true);
    try {
      const job = await jobsApi.createJob(kind, payload);
      if (!current()) return;
      addJob(job); setJobs((current) => [job, ...current.filter((item) => item.id !== job.id)]);
      if (kind !== 'dataset.download') setView('results');
      toast.success(kind === 'dataset.download' ? tr('官方集合下载已提交', 'Official collection download submitted') : tr('评测任务已提交', 'Evaluation job submitted'));
    } catch (cause) { if (current()) toast.error(errorMessage(cause)); }
    finally { if (current()) setBusy(false); }
  }
  async function cancel(id: string) {
    const current = connectionScope();
    try {
      const next = await jobsApi.cancelJob(id);
      if (current()) setJobs((jobs) => jobs.map((item) => item.id === next.id ? next : item));
    } catch (cause) { if (current()) toast.error(errorMessage(cause)); }
  }
  async function compare() {
    if (busy) return;
    const current = connectionScope();
    setBusy(true);
    try {
      const next = await evaluationsApi.compareEvaluations(selected);
      if (current()) setComparison(next);
    } catch (cause) { if (current()) toast.error(errorMessage(cause)); }
    finally { if (current()) setBusy(false); }
  }
  const selectedResults = evaluations.filter((item) => selected.includes(item.id));
  const officialResult = (item: EvaluationResult) => !['perplexity', 'accuracy_benchmark'].includes(item.kind) || officialCatalog.some((spec) =>
    spec.sha256 === (item.dataset_manifest.source_sha256 || item.dataset_manifest.sha256) && spec.byte_size === (item.dataset_manifest.source_byte_size || item.dataset_manifest.byte_size));
  const comparable = selectedResults.length >= 2 && selectedResults.length <= 16 &&
    selectedResults.every((item) => officialResult(item) && item.kind === selectedResults[0].kind && item.comparison_key === selectedResults[0].comparison_key);
  const catalog = [
    { id: 'quality', name: 'WT2 KLD / Top1', description: tr('量化质量', 'Quantization quality'), icon: 'flask', available: tools?.quality_available },
    { id: 'performance', name: tr('推理测速', 'Inference benchmark'), description: 'prefill · decode · MTP', icon: 'gauge', available: tools?.benchmark_available },
  ] as const;
  const task = taskBenchmarks.find((item) => item.id === tab);
  const catalogButton = (item: { id: typeof tab; name: string; description: string; icon?: 'flask' | 'gauge'; mark?: TaskBenchmark['id']; available: boolean | undefined }) =>
    <button key={item.id} type="button" role="tab" aria-label={item.name} aria-selected={tab === item.id} onClick={() => setTab(item.id)}>
      {item.mark ? <BenchmarkMark id={item.mark} /> : item.icon && <Icon name={item.icon} size={17} />}<div><strong>{item.name}</strong><span>{item.description}</span></div>{!item.available && <small>{tr('工具不可用', 'Tools unavailable')}</small>}
    </button>;
  return <div className="evaluation-workbench">
    <section className="tm-panel evaluation-topbar">
      <div className="panel-heading"><div><h2>{tr('测评平台', 'Benchmark platform')}</h2><p>{tr('选择测试、配置条件，统一管理任务与结果', 'Choose tests, configure conditions, manage jobs and results together')}</p></div>
        <button className="panel-action" onClick={() => void refreshResults()} type="button">{tr('刷新结果', 'Refresh results')}</button></div>
      <nav className="evaluation-navigation" aria-label={tr('测评平台导航', 'Benchmark platform navigation')}>{([
        ['configure', tr('测试配置', 'Test setup')], ['datasets', tr('官方集合', 'Official collections')], ['results', tr('任务与结果', 'Jobs & results')],
      ] as const).map(([id, label]) => <button type="button" key={id} aria-current={view === id ? 'page' : undefined} onClick={() => setView(id)}>{label}{id === 'results' && hasActiveJobs && <span className="evaluation-running-dot" />}</button>)}</nav>
    </section>
    <section className="tm-panel evaluation-setup" hidden={view !== 'configure'}>
      <div className="evaluation-catalog" role="tablist" aria-label={tr('测试目录', 'Test catalog')}>
        <div className="evaluation-catalog-group"><h3>{tr('基础', 'Basic')}</h3>{catalog.map(catalogButton)}</div>
        <div className="evaluation-catalog-group"><h3>{tr('任务 benchmark', 'Task benchmarks')}</h3>{taskBenchmarkGroups.map((group) =>
          <section className="evaluation-catalog-subgroup" key={group.id} aria-label={tr(group.zh, group.en)}><h4>{tr(group.zh, group.en)}<BenchmarkGroupMark group={group.id} /></h4>{taskBenchmarks.filter((item) => item.group === group.id).map((item) => catalogButton({
            id: item.id, name: item.name, description: tr(item.zh, item.en), mark: item.id,
            available: tools?.task_benchmarks?.[item.dataset]?.available,
          }))}</section>)}</div>
      </div>
      <div className="evaluation-configuration"><div className="evaluation-config-heading"><span>{task?.name || tr('测试条件', 'Test conditions')}</span><small>{task ? tr('官方评分', 'Official scoring') : tab === 'quality' ? 'WT2 · reference logits' : 'MFQ · native timing'}</small></div>
      {tab === 'quality' && <QualityForm models={models} datasets={datasets} instances={instances} available={!!tools?.quality_available} referenceAvailable={!!tools?.reference_available} outputRoot={tools?.workspace_root || ''} generatedReference={generatedReference} busy={busy} submit={submit} tr={tr} />}
      {tab === 'performance' && <PerformanceForm instances={instances} available={!!tools?.benchmark_available} busy={busy} submit={submit} tr={tr} />}
      {task && <AccuracyForm key={task.id} task={task} instances={instances} datasets={datasets} available={!!tools?.task_benchmarks?.[task.dataset]?.available} readiness={tools?.task_benchmarks?.[task.dataset]} busy={busy} submit={submit} tr={tr} />}
      </div>
    </section>
    <div hidden={view !== 'datasets'}><OfficialDatasets catalog={officialCatalog} datasets={datasets} jobs={jobs} busy={busy} readiness={tools?.task_benchmarks} submit={submit} tr={tr} /></div>
    <div className="evaluation-results-view" hidden={view !== 'results'}>
    {jobs.length > 0 && <section className="tm-panel"><details open={hasActiveJobs}><summary className="evaluation-job-heading">{tr('评测任务', 'Evaluation jobs')}<small>{tr(`${jobs.filter(activeJob).length} 个正在运行`, `${jobs.filter(activeJob).length} active`)}</small></summary>
      <div className="evaluation-job-list">{jobs.slice(0, 8).map((job) => <div key={job.id}>
        <div><strong>{job.kind === 'benchmark.inference' ? tr('推理测速', 'Inference benchmark') : job.kind === 'evaluate.accuracy' ? String(datasets.find((item) => item.id === job.payload.dataset_id)?.metadata.task || tr('任务 benchmark', 'Task benchmark')) : job.kind === 'dataset.download' ? tr('官方集合下载', 'Official collection download') : job.kind === 'reference.wikitext2' ? tr('WT2 logits 生成', 'WT2 logits generation') : 'WT2 KLD / Top1'}</strong><small>{statusLabels[job.status]} · {(job.progress * 100).toFixed(0)}%</small></div>
        <progress aria-label={tr('评测进度', 'Evaluation progress')} max={1} value={job.progress} />
        {activeJob(job) && job.progress_data?.source_loading?.fallback_reason === 'insufficient_memory' && <p role="status">{tr('空余内存不足，正在逐层生成。', 'Insufficient free memory; generating layerwise.')}</p>}
        {activeJob(job) && <button type="button" disabled={job.status === 'cancelling'} onClick={() => void cancel(job.id)}>{tr('取消', 'Cancel')}</button>}
        {job.error && <p role="status">{job.error.message}</p>}
        {job.kind === 'reference.wikitext2' && job.status === 'succeeded' && typeof job.result?.output === 'string' && typeof job.result.manifest === 'string' && <div className="evaluation-reference-result">
          <p>{job.result.output}</p><p>{job.result.manifest}</p>
          {!!job.result.loading && typeof job.result.loading === 'object' && 'fallback_reason' in job.result.loading && job.result.loading.fallback_reason === 'insufficient_memory' && <p role="status">{tr('启动时空余内存不足，已自动改为逐层生成。', 'Free memory was insufficient at startup; generation automatically used layerwise loading.')}</p>}
          <button type="button" onClick={() => {
            setGeneratedReference({ id: job.id, output: String(job.result!.output), manifest: String(job.result!.manifest), context_size: Number(job.result!.context_size), chunks: Number(job.result!.chunks), parallel: 1, dataset_id: String(job.result!.dataset_id) });
            setTab('quality'); setView('configure');
          }}>{tr('用于 WT2 评测', 'Use for WT2 evaluation')}</button>
        </div>}
      </div>)}</div></details>
    </section>}
    <section className="tm-panel evaluation-history"><div className="panel-heading"><div><h2>{tr('历史结果', 'Result history')}</h2><p>{tr('保存测试条件、实际 token 数与每轮原始数据；仅对比相同条件的结果', 'Conditions, actual token counts and raw rounds are saved; compare only matching conditions')}</p></div><b>{evaluations.length}</b></div>
      {!evaluations.length && <div className="inline-empty">{tr('选择上方测试并运行，结果会自动保存在这里', 'Choose and run a test above; results are saved here automatically')}</div>}
      {evaluations.length > 0 && <label className="evaluation-answer-filter">{tr('结果类型', 'Result type')}<select value={resultKind} onChange={(event) => setResultKind(event.target.value)}><option value="all">{tr('全部', 'All')}</option><option value="perplexity">WT2 KLD / Top1</option><option value="inference_benchmark">{tr('推理测速', 'Inference benchmark')}</option><option value="accuracy_benchmark">{tr('答案评测', 'Answer evaluation')}</option></select></label>}
      <div className="evaluation-results">{evaluations.filter((item) => resultKind === 'all' || item.kind === resultKind).map((item) => <article key={item.id}>
        <header><label><input type="checkbox" checked={selected.includes(item.id)} onChange={(event) => { setComparison(null); setSelected((current) => event.target.checked ? [...current, item.id] : current.filter((id) => id !== item.id)); }} />
          <strong>{item.model_id}</strong></label><ModelVendorMark name={item.model_id} /></header>
        <small>{item.kind === 'inference_benchmark' ? tr('推理测速', 'Inference benchmark') : item.kind === 'accuracy_benchmark' ? String(item.parameters.benchmark || tr('答案评测', 'Answer evaluation')) : item.kind === 'perplexity' ? 'WT2 / KLD' : item.kind} · {new Date(item.created_at).toLocaleString()}</small>
        {!officialResult(item) && <p className="evaluation-note">{tr('旧测试 · 未验证官方集合，不参与标准结果对比', 'Legacy test · unverified collection, excluded from standard comparisons')}</p>}
        {item.kind === 'inference_benchmark' && resultObjects(item.metrics.series).length > 0 ? <ThroughputTable series={item.metrics.series} tr={tr} /> : <dl className="evaluation-metrics">{metricKeys.filter((key) => key in item.metrics).map((key) => <div key={key}><dt>{'pass_at_1' in item.metrics && key === 'accuracy' ? 'pass@1' : 'pass_at_1' in item.metrics && key === 'sample_count' ? tr('实测样本', 'Samples tested') : 'pass_at_1' in item.metrics && key === 'correct' ? tr('通过样本', 'Passing samples') : metricLabels[key]}</dt><dd>{metricValue(key, item.metrics[key])}</dd></div>)}</dl>}
        {item.kind === 'accuracy_benchmark' && <AnswerDetails metrics={item.metrics} tr={tr} />}
        <details className="evaluation-result-details"><summary>{tr('条件与逐轮数据', 'Conditions & raw rounds')}</summary><pre>{JSON.stringify({ parameters: item.parameters, metrics: item.metrics, dataset: item.dataset_manifest, runtime: item.runtime_identity }, null, 2)}</pre></details>
        <footer className="evaluation-result-footer"><button className="evaluation-export" type="button" onClick={() => exportEvaluation(item)}>{tr('导出 JSON', 'Export JSON')}</button></footer>
      </article>)}</div>
      <button className="panel-action" type="button" disabled={busy || !comparable} onClick={() => void compare()}>{tr('对比所选结果', 'Compare selected')}</button>
      {selected.length >= 2 && !comparable && <p className="evaluation-note">{tr('所选结果的测试条件不同，不能直接对比', 'Selected results have different conditions and cannot be compared directly')}</p>}
      {comparison && <div className="evaluation-comparison"><table><thead><tr><th>{tr('模型', 'Model')}</th><th>{tr('输入 token', 'Input tokens')}</th>{comparison.metrics.filter((key) => metricKeys.includes(key as typeof metricKeys[number])).map((key) => <th key={key}>{metricLabels[key]}</th>)}</tr></thead>
        <tbody>{comparison.rows.flatMap((row) => {
          const series = resultObjects(row.evaluation.metrics.series);
          return (series.length ? series : [row.evaluation.metrics]).map((metrics, index) => <tr key={`${row.evaluation.id}-${index}`}><th>{row.evaluation.model_id}</th><td>{String(metrics.prompt_tokens ?? '—')}</td>{comparison.metrics.filter((key) => metricKeys.includes(key as typeof metricKeys[number])).map((key) => <td key={key}>{metricValue(key, metrics[key])}</td>)}</tr>);
        })}</tbody></table></div>}
    </section>
    </div>
  </div>;
}
