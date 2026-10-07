import { useEffect, useState, type FormEvent } from 'react';
import type { BenchmarkParameters, DatasetResource, ModelArtifact, OfficialBenchmarkReadiness, RuntimeInstance } from '../../shared/api/types';
import type { TaskBenchmark } from './benchmarkTasks';
import { Wt2ReferenceGenerator, type GeneratedWt2Reference } from './Wt2ReferenceGenerator';

type Translate = (zh: string, en: string) => string;
type Submit = (kind: string, payload: Record<string, unknown>) => Promise<void>;

export function QualityForm({ models, datasets, instances, available, referenceAvailable = false, outputRoot = '', generatedReference, busy, submit, tr }: {
  models: ModelArtifact[]; datasets: DatasetResource[]; instances: RuntimeInstance[]; available: boolean; referenceAvailable?: boolean;
  outputRoot?: string; generatedReference?: GeneratedWt2Reference | null; busy: boolean; submit: Submit; tr: Translate;
}) {
  const [model, setModel] = useState('');
  const [dataset, setDataset] = useState('');
  const [reference, setReference] = useState('');
  const [manifest, setManifest] = useState('');
  const [context, setContext] = useState(512);
  const [chunks, setChunks] = useState(8);
  const [parallel, setParallel] = useState(1);
  const wt2 = datasets.filter((item) => item.kind === 'wikitext2');
  const alreadyLoaded = instances.some((item) => item.model === model && ['loading', 'ready', 'busy'].includes(item.state));
  useEffect(() => { if (!models.some((item) => item.name === model)) setModel(models[0]?.name || ''); }, [models, model]);
  useEffect(() => { if (!datasets.some((item) => item.id === dataset)) setDataset(datasets.find((item) => item.kind === 'wikitext2')?.id || ''); }, [datasets, dataset]);
  useEffect(() => {
    if (!generatedReference) return;
    setReference(generatedReference.output); setManifest(generatedReference.manifest);
    setContext(generatedReference.context_size); setChunks(generatedReference.chunks);
    setParallel(generatedReference.parallel); setDataset(generatedReference.dataset_id);
  }, [generatedReference]);
  async function run(event: FormEvent) {
    event.preventDefault();
    await submit('evaluate.wikitext2', { model, dataset_id: dataset, reference_logits: reference.trim(),
      reference_manifest: manifest.trim(), context_size: context, chunks, parallel });
  }
  return <form className="evaluation-form" onSubmit={(event) => void run(event)}>
    <fieldset className="evaluation-fieldset"><legend>{tr('模型与集合', 'Model & collection')}</legend><div className="evaluation-fields evaluation-fields-pair">
      <label className="evaluation-wide">{tr('待测模型', 'Candidate model')}<select className="evaluation-primary-select" value={model} onChange={(event) => setModel(event.target.value)} required>
        {!models.length && <option value="">{tr('没有完整模型资产', 'No complete model assets')}</option>}
        {models.map((item) => <option key={item.id} value={item.name}>{item.name}</option>)}
      </select></label>
      <label className="evaluation-wide">{tr('WT2 测试集', 'WT2 test dataset')}<select className="evaluation-primary-select" value={dataset} onChange={(event) => setDataset(event.target.value)} required>
        {!wt2.length && <option value="">{tr('请先下载官方 WT2 集合', 'Download the official WT2 collection first')}</option>}
        {wt2.map((item) => <option key={item.id} value={item.id}>{item.name} · {item.sha256.slice(0, 12)}</option>)}
      </select></label>
    </div></fieldset>
    <fieldset className="evaluation-fieldset"><legend>{tr('参考分布', 'Reference distribution')}</legend><div className="evaluation-fields evaluation-fields-pair">
      <label>{tr('参考 logits 文件', 'Reference logits file')}<input value={reference} onChange={(event) => setReference(event.target.value)} placeholder="datasets/wt2-reference.logits" required /></label>
      <label>{tr('参考 manifest', 'Reference manifest')}<input value={manifest} onChange={(event) => setManifest(event.target.value)} placeholder="datasets/wt2-reference.logits.manifest.json" required /></label>
    </div></fieldset>
    {referenceAvailable && <Wt2ReferenceGenerator dataset={dataset} contextSize={context} chunks={chunks} available={referenceAvailable} busy={busy} outputRoot={outputRoot} submit={submit} />}
    <fieldset className="evaluation-fieldset"><legend>{tr('测试条件', 'Test conditions')}</legend><div className="evaluation-fields">
      <label>ctx<input type="number" min={32} max={1048576} value={context} onChange={(event) => setContext(Number(event.target.value))} required /></label>
      <label>{tr('测试窗口数', 'Context windows')}<input type="number" min={1} max={10000} value={chunks} onChange={(event) => setChunks(Number(event.target.value))} required /></label>
      <label>{tr('并行序列', 'Parallel sequences')}<input type="number" min={1} max={64} value={parallel} onChange={(event) => setParallel(Number(event.target.value))} required /></label>
    </div></fieldset>
    <details className="evaluation-advanced"><summary>{tr('参考分布与计分协议', 'Reference distribution & scoring protocol')}</summary><p className="evaluation-note">{tr('KLD = KL(参考 ∥ 待测)，Top1 为参考首选 token 一致率。只接受官方 WT2 完整文件，校验后按 mfq-wt2-text-v1 生成固定文本。参考 logits 的 tokenizer、ctx 与并行数必须一致。Flash-Next 的并行序列暂限 1。', 'KLD = KL(reference ∥ candidate); Top1 is reference top-token agreement. Only the official complete WT2 file is accepted; mfq-wt2-text-v1 produces a fixed text corpus after verification. Reference logits must match the tokenizer, ctx and parallel sequences. Flash-Next currently requires one sequence.')}</p><p className="evaluation-note">{tr('参考文本 SHA256', 'Reference text SHA256')}: <code>7952b062817620ac99b47296dc945942b55068d0e85baf1f95f300c7ad1480ef</code></p></details>
    {!available && <p role="status" className="evaluation-note">{tr('后端尚未提供 WT2 质量评测工具', 'WT2 quality tools are unavailable on this backend')}</p>}
    {alreadyLoaded && <p role="status" className="evaluation-note">{tr('待测模型已在服务中加载，请先卸载，避免质量评测重复载入权重。', 'The candidate is loaded in the service. Unload it first to avoid loading its weights twice for quality evaluation.')}</p>}
    <div className="evaluation-run-footer"><span>{tr(`${chunks} 个窗口 · ctx ${context} · ${parallel} 条并行序列`, `${chunks} windows · ctx ${context} · ${parallel} parallel sequences`)}</span><button className="evaluation-run" type="submit" disabled={busy || !available || alreadyLoaded || !model || !dataset || !reference.trim() || !manifest.trim()}>{tr('运行 WT2 KLD / Top1', 'Run WT2 KLD / Top1')}</button></div>
  </form>;
}

export const DEFAULT_BENCHMARK_PROMPT = 'A small engineering team is designing an efficient inference service. They need to balance memory use, throughput, latency, and reliability. Explain the practical trade-offs with concrete examples, then continue with a detailed implementation plan, measurement methods, and failure recovery. Discuss why reproducible inputs and clearly defined timing boundaries matter. Continue the explanation naturally and do not summarize prematurely. ';

export function PerformanceForm({ instances, available, busy, submit, tr }: {
  instances: RuntimeInstance[]; available: boolean; busy: boolean; submit: Submit; tr: Translate;
}) {
  const [instanceId, setInstanceId] = useState('');
  const [inputLengths, setInputLengths] = useState('4096');
  const [draft, setDraft] = useState({ prompt: DEFAULT_BENCHMARK_PROMPT, prompt_tokens: 4096, output_tokens: 128,
    mode: 'both', repetitions: 3, warmup_runs: 1, cooldown_seconds: 0, temperature: 0,
    top_k: 20, top_p: 0.95, seed: 42, mtp_max_draft_tokens: 3 });
  const loaded = instances.filter((item) => item.state === 'ready' || item.state === 'busy');
  useEffect(() => { if (!instances.some((item) => item.id === instanceId)) setInstanceId(instances.find((item) => item.state === 'ready')?.id || ''); }, [instances, instanceId]);
  const selected = loaded.find((item) => item.id === instanceId);
  const lengths = inputLengths.split(/[,，\s]+/).filter(Boolean).map(Number);
  const validLengths = lengths.length > 0 && lengths.length <= 8 && new Set(lengths).size === lengths.length && lengths.every((value) => Number.isInteger(value) && value >= 1 && value <= 1048576);
  const exceedsContext = !!selected?.context_size && Math.max(...lengths) + draft.output_tokens > selected.context_size;
  const unavailableMtp = draft.mode !== 'decode' && selected && !selected.mtp_available;
  const fields = [
    ['output_tokens', tr('输出 token 上限', 'Output token limit'), 2, 8192, 1],
    ['repetitions', tr('实测轮数', 'Measured rounds'), 1, 20, 1],
    ['warmup_runs', tr('每种模式暖机轮数', 'Warmup rounds per mode'), 0, 3, 1],
    ['cooldown_seconds', tr('轮间冷却（秒）', 'Cooldown between runs (s)'), 0, 60, 1],
    ['temperature', tr('温度', 'Temperature'), 0, 10, 0.1],
    ['top_k', 'Top K', 0, 1024, 1], ['top_p', 'Top P', 0.01, 1, 0.01],
    ['seed', tr('随机种子', 'Seed'), 0, 2147483647, 1],
    ['mtp_max_draft_tokens', tr('MTP 草稿深度上限', 'MTP maximum draft depth'), 1, 5, 1],
  ] as const;
  const numberField = ([name, label, min, max, step]: typeof fields[number]) => <label key={name}>{label}<input type="number" min={min} max={max} step={step} value={draft[name]} required onChange={(event) => setDraft({ ...draft, [name]: Number(event.target.value) })} /></label>;
  function toggleLength(length: number) {
    const next = lengths.includes(length) ? lengths.filter((value) => value !== length) : [...lengths, length];
    setInputLengths([...new Set(next)].sort((a, b) => a - b).join(', '));
  }
  return <form className="evaluation-form" onSubmit={(event) => { event.preventDefault(); if (validLengths && !exceedsContext) void submit('benchmark.inference', { ...draft, prompt_tokens: lengths[0], prompt_token_lengths: lengths, instance_id: instanceId }); }}>
    <fieldset className="evaluation-fieldset"><legend>{tr('待测模型', 'Candidate model')}</legend><div className="evaluation-fields">
      <label className="evaluation-wide">{tr('已加载模型', 'Loaded model')}<select className="evaluation-primary-select" value={instanceId} onChange={(event) => setInstanceId(event.target.value)} required>
        {!loaded.length && <option value="">{tr('请先加载一个模型', 'Load a model first')}</option>}
        {loaded.map((item) => <option key={item.id} value={item.id}>{item.model} · ctx {item.context_size?.toLocaleString()} · {item.mtp_available ? 'MTP' : tr('无 MTP', 'No MTP')}</option>)}
      </select></label>
    </div></fieldset>
    <fieldset className="evaluation-fieldset"><legend>{tr('测试条件', 'Test conditions')}</legend><div className="evaluation-fields evaluation-fields-mode">
      <label>{tr('测试模式', 'Test mode')}<select value={draft.mode} onChange={(event) => setDraft({ ...draft, mode: event.target.value })}>
        <option value="both">{tr('普通 decode + MTP 对照', 'Ordinary decode + MTP comparison')}</option>
        <option value="decode">{tr('普通 decode', 'Ordinary decode')}</option><option value="mtp">MTP</option>
      </select></label>
      {fields.slice(0, 2).map(numberField)}
    </div></fieldset>
    <fieldset className="evaluation-fieldset"><legend>{tr('输入规模', 'Input sizes')}</legend><div className="evaluation-fields">
      <label className="evaluation-wide">{tr('输入 token 长度', 'Input token lengths')}<input value={inputLengths} onChange={(event) => setInputLengths(event.target.value)} placeholder="1024, 4096, 8192" required /></label>
      <div className="evaluation-lengths evaluation-wide">{[1024, 4096, 8192, 16384, 32768, 65536].map((length) => <label key={length}><input type="checkbox" checked={lengths.includes(length)} onChange={() => toggleLength(length)} />{length / 1024}k</label>)}</div>
    </div></fieldset>
    <details className="evaluation-advanced"><summary>{tr('输入文本与高级条件', 'Input text & advanced conditions')}</summary><div className="evaluation-fields">
      {fields.slice(2).map(numberField)}
      <label className="evaluation-wide">{tr('输入文本', 'Input text')}<textarea value={draft.prompt} onChange={(event) => setDraft({ ...draft, prompt: event.target.value })} maxLength={262144} required rows={4} /></label>
    </div></details>
    <details className="evaluation-advanced"><summary>{tr('计时协议', 'Timing protocol')}</summary><p className="evaluation-note">{tr('按模型 tokenizer 精确调整输入长度：不足时重复、超长时截断。每轮重新 prefill，不读写前缀缓存；暖机不计入汇总。decode 从首 token 计时到末 token，包含调度和读取等待。临时独占待测模型，不改服务 MTP 开关。', 'Input lengths are exact in the model tokenizer: repeated if short, truncated if long. Every run prefills anew, without prefix-cache reads or writes; warmups are excluded. Decode spans the first to last token, including scheduling and I/O waits. The candidate is exclusive; the service MTP switch is unchanged.')}</p></details>
    {exceedsContext && <p role="status" className="evaluation-note">{tr('输入加输出超出已加载 ctx，请先调整模型上下文并重载', 'Input plus output exceeds the loaded ctx; reload the model with a larger context first')}</p>}
    {!validLengths && <p role="status" className="evaluation-note">{tr('请输入 1–8 个互不重复的 token 长度，以逗号分隔', 'Enter 1–8 distinct token lengths, separated by commas')}</p>}
    {unavailableMtp && <p role="status" className="evaluation-note">{tr('此实例没有可用 MTP，请选择普通 decode', 'This instance has no usable MTP; choose ordinary decode')}</p>}
    {!available && <p role="status" className="evaluation-note">{tr('后端尚未提供推理测速任务', 'Inference benchmarking is unavailable on this backend')}</p>}
    <div className="evaluation-run-footer"><span>{tr(`${validLengths ? lengths.length : '—'} 档输入 · ${draft.mode === 'both' ? 2 : 1} 种模式 · ${draft.repetitions} 轮`, `${validLengths ? lengths.length : '—'} lengths · ${draft.mode === 'both' ? 2 : 1} modes · ${draft.repetitions} rounds`)}</span><button className="evaluation-run" type="submit" disabled={busy || !available || selected?.state !== 'ready' || !validLengths || exceedsContext || !!unavailableMtp || !draft.prompt.trim()}>{tr('开始测速', 'Start benchmark')}</button></div>
  </form>;
}

export function AccuracyForm({ task, instances, datasets, available, readiness, busy, submit, tr }: {
  task: TaskBenchmark; instances: RuntimeInstance[]; datasets: DatasetResource[]; available: boolean;
  readiness?: OfficialBenchmarkReadiness;
  busy: boolean; submit: Submit; tr: Translate;
}) {
  const [instanceId, setInstanceId] = useState('');
  const [datasetId, setDatasetId] = useState('');
  const [officialDefaults, setOfficialDefaults] = useState(true);
  const [draft, setDraft] = useState<BenchmarkParameters>({ protocol: task.protocol, sample_count: task.samples, max_tokens: task.tokens, seed: 42,
    enable_thinking: task.protocol === 'math', enable_mtp: false, temperature: 0, top_p: 1, top_k: 0, num_generations: 1 });
  const defaults = readiness?.defaults;
  const settings = officialDefaults ? { ...draft, ...defaults?.parameters } : draft;
  const missingDefaults = officialDefaults && !defaults?.available;
  const numberValue = (name: 'max_tokens' | 'seed' | 'temperature' | 'top_p' | 'top_k' | 'num_generations') => missingDefaults ? '' : settings[name] ?? '';
  const loaded = instances.filter((item) => item.state === 'ready' || item.state === 'busy');
  const custom = datasets.filter((item) => item.metadata?.official_id === task.dataset);
  useEffect(() => { if (!instances.some((item) => item.id === instanceId)) setInstanceId(instances.find((item) => item.state === 'ready')?.id || ''); }, [instances, instanceId]);
  useEffect(() => { if (!custom.some((item) => item.id === datasetId)) setDatasetId(custom[0]?.id || ''); }, [datasets, datasetId, task.dataset]);
  const instance = loaded.find((item) => item.id === instanceId);
  const protocolNames: Record<string, string> = {
    'official-api-cot': tr('官方 API · CoT 示例', 'Official API · CoT examples'),
    'official-zero-shot-sample': tr('官方 zero-shot · sample', 'Official zero-shot · sample'),
    'official-5shot-likelihood': tr('官方 5-shot · 概率评分', 'Official 5-shot · likelihood scoring'),
    'official-mc-likelihood': tr('官方 MC · 概率评分', 'Official MC · likelihood scoring'),
    'official-v6-python-pass1': 'LiveCodeBench v6 · Python pass@1',
    'official-matharena-aime2025': 'MathArena · AIME2025',
  };
  const protocolName = readiness ? protocolNames[readiness.protocol] || readiness.protocol : tr('等待官方评分信息', 'Waiting for official scoring details');
  const unavailableReason = readiness?.reason === 'likelihood_unavailable' ? tr('官方协议所需的概率评分接口尚未接入，不会用字母答案评分代替。', 'The likelihood interface required by the official protocol is not integrated; answer-letter scoring is not substituted.')
    : readiness?.reason === 'official_scorer_missing' ? tr('请在官方集合中下载或校验此任务，同时安装固定版本的官方评分代码。', 'Download or verify this task in Official collections to install its pinned official scoring code.')
    : readiness?.reason === 'official_dependencies_missing' ? tr('官方评分依赖尚未安装。', 'Official scoring dependencies are not installed.')
    : tr('此任务的官方执行器尚未接齐，暂不能运行；不使用近似评分。', 'This task’s official runner is not fully integrated and cannot run yet; approximate scoring is disabled.');
  return <form className="evaluation-form" onSubmit={(event) => { event.preventDefault(); if (!missingDefaults) void submit('evaluate.accuracy', { ...settings, official_defaults: officialDefaults, ...(task.protocol === 'likelihood' ? { max_tokens: 1, enable_thinking: false, enable_mtp: false } : {}), instance_id: instanceId, dataset_id: datasetId }); }}>
    <fieldset className="evaluation-fieldset"><legend>{tr('模型与集合', 'Model & collection')}</legend><div className="evaluation-fields">
      <label className="evaluation-wide">{tr('已加载模型', 'Loaded model')}<select className="evaluation-primary-select" value={instanceId} onChange={(event) => setInstanceId(event.target.value)} required>{!loaded.length && <option value="">{tr('请先加载一个模型', 'Load a model first')}</option>}{loaded.map((item) => <option key={item.id} value={item.id}>{item.model}</option>)}</select></label>
      <label className="evaluation-wide">{tr('官方测试集', 'Official test dataset')}<select className="evaluation-primary-select" value={datasetId} onChange={(event) => setDatasetId(event.target.value)} required>{!custom.length && <option value="">{tr('请先从官方集合下载此任务的数据', 'Download this task from Official collections first')}</option>}{custom.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}</select></label>
    </div></fieldset>
    <div className="evaluation-default-toggle"><label><input type="checkbox" checked={officialDefaults} onChange={(event) => setOfficialDefaults(event.target.checked)} />{tr('官方默认', 'Official defaults')}</label><small>{officialDefaults ? tr('采用固定版本官方仓库配置', 'Uses the pinned official repository configuration') : tr('自定义参数', 'Custom parameters')}</small></div>
    <fieldset className="evaluation-fieldset" disabled={officialDefaults}><legend>{tr('测试条件', 'Test conditions')}</legend><div className="evaluation-fields">
      <label className="evaluation-wide">{tr('评分协议', 'Scoring protocol')}<input value={protocolName} readOnly /></label>
      <label>{tr('抽样题数（0 = 全量）', 'Sample count (0 = all)')}<input type="number" min={0} max={100000} value={officialDefaults ? defaults?.parameters.sample_count ?? '' : draft.sample_count} onChange={(event) => setDraft({ ...draft, sample_count: Number(event.target.value) })} required /></label>
      {task.protocol !== 'likelihood' && <label>{tr('输出 token 上限', 'Output token limit')}<input type="number" min={1} max={8192} value={numberValue('max_tokens')} onChange={(event) => setDraft({ ...draft, max_tokens: event.target.value === '' ? null : Number(event.target.value) })} placeholder={tr('模型 / 架构默认', 'Model / architecture default')} /></label>}
      <label>{task.protocol === 'likelihood' ? tr('抽样种子', 'Sampling seed') : tr('抽样与生成种子', 'Sampling & generation seed')}<input type="number" min={0} value={numberValue('seed')} onChange={(event) => setDraft({ ...draft, seed: event.target.value === '' ? null : Number(event.target.value) })} placeholder={tr('官方未设定', 'Not specified officially')} /></label>
      {task.protocol !== 'likelihood' && <><label>{tr('温度', 'Temperature')}<input type="number" min={0} max={10} step={.1} value={numberValue('temperature')} onChange={(event) => setDraft({ ...draft, temperature: event.target.value === '' ? null : Number(event.target.value) })} placeholder={tr('模型 / 架构默认', 'Model / architecture default')} /></label>
        <label>Top P<input type="number" min={.01} max={1} step={.01} value={numberValue('top_p')} onChange={(event) => setDraft({ ...draft, top_p: event.target.value === '' ? null : Number(event.target.value) })} placeholder={tr('模型 / 架构默认', 'Model / architecture default')} /></label>
        <label>Top K<input type="number" min={0} max={1024} value={numberValue('top_k')} onChange={(event) => setDraft({ ...draft, top_k: event.target.value === '' ? null : Number(event.target.value) })} placeholder={tr('模型 / 架构默认', 'Model / architecture default')} /></label>
        <div className="evaluation-checks evaluation-wide"><label><input type="checkbox" ref={(input) => { if (input) input.indeterminate = settings.enable_thinking === null; }} checked={!missingDefaults && !!settings.enable_thinking} onChange={(event) => setDraft({ ...draft, enable_thinking: event.target.checked })} />{settings.enable_thinking === null ? tr('思考：模型 / 架构默认', 'Thinking: model / architecture default') : tr('启用思考', 'Enable thinking')}</label><label><input type="checkbox" disabled={!instance?.mtp_available} checked={!missingDefaults && settings.enable_mtp} onChange={(event) => setDraft({ ...draft, enable_mtp: event.target.checked })} />MTP</label></div></>}
      {task.protocol === 'code' && <label>{tr('每题生成次数', 'Generations per question')}<input type="number" min={1} max={20} value={numberValue('num_generations')} onChange={(event) => setDraft({ ...draft, num_generations: Number(event.target.value) })} required /></label>}
    </div></fieldset>
    {task.protocol !== 'likelihood' && <p className="evaluation-note">{tr('空代表官方无默认值，自动读取模型/架构默认值', 'Empty means no official default; model / architecture defaults are read automatically')}</p>}
    <details className="evaluation-advanced"><summary>{tr('抽样与评分协议', 'Sampling & scoring protocol')}</summary>
      <p className="evaluation-note">{tr('使用官方仓库原始评分函数；数据文件与评分代码均固定版本、校验完整 SHA256。保留官方的答案解析、失败处理及汇总规则，不使用自定义正则或近似评分替代。不同评分版本不能直接对比。', 'Uses original scoring functions from the official repository. Dataset files and scoring code are revision-pinned and whole-file SHA256 verified. Official parsing, failure handling and aggregation rules are retained, without custom regex or approximate substitutes. Different scoring versions are not directly comparable.')}</p>
      <p className="evaluation-note"><a href={'https://github.com/' + task.repository + (readiness ? '/tree/' + readiness.revision : '')} target="_blank" rel="noreferrer">{task.repository}</a>{readiness && <code> · {readiness.revision.slice(0, 12)}</code>}</p>
      {readiness && <pre>{JSON.stringify(readiness.scoring_files, null, 2)}</pre>}
      {task.id === 'accuracy' && <p className="evaluation-note">{tr('MMLU-Pro 使用官方 validation CoT 示例、原始答案解析以及官方解析失败时的随机计分规则；结果会标记该回退，和旧零样本字母一致率结果分开。', 'MMLU-Pro uses the official validation CoT examples, original answer extraction and official random scoring fallback for unparsed answers. The fallback is recorded; these results are separate from legacy zero-shot answer-letter agreement.')}</p>}
      {task.protocol === 'code' && <p className="evaluation-note">{tr('固定 v6 独立批次，不是累计 release_v6；使用官方 Python 测试与 pass@1 汇总。', 'Pinned standalone v6 batch, not cumulative release_v6; uses official Python tests and pass@1 aggregation.')}</p>}
      {task.protocol === 'math' && <p className="evaluation-note">{tr('使用 MathArena 发布的 AIME I + II 共 30 题及对应评分实现；源文件的 train 仅是发布者划分名称。', 'Uses all 30 AIME I + II problems released by MathArena and its corresponding scoring implementation. The source split train is the publisher’s split name.')}</p>}
    </details>
    {!available && <p role="status" className="evaluation-note">{unavailableReason}</p>}
    {missingDefaults && <p role="status" className="evaluation-note">{tr('正在读取官方默认配置', 'Loading official defaults')}</p>}
    <div className="evaluation-run-footer"><span>{missingDefaults ? '—' : task.protocol === 'likelihood' ? tr(`${settings.sample_count || '全部'} 道题 · 原始文本概率`, `${settings.sample_count || 'All'} questions · raw-text probabilities`) : tr(`${settings.sample_count || '全部'} 道题 · ${settings.max_tokens == null ? '模型默认输出上限' : `最多 ${settings.max_tokens} tokens`}${settings.num_generations > 1 ? ` · 每题 ${settings.num_generations} 次` : ''}`, `${settings.sample_count || 'All'} questions · ${settings.max_tokens == null ? 'model-default output limit' : `up to ${settings.max_tokens} tokens`}${settings.num_generations > 1 ? ` · ${settings.num_generations} generations each` : ''}`)}</span><button className="evaluation-run" type="submit" disabled={busy || !available || missingDefaults || instance?.state !== 'ready' || !datasetId || (settings.enable_mtp && !instance?.mtp_available) || (!!instance?.context_size && settings.max_tokens != null && settings.max_tokens >= instance.context_size)}>{task.protocol === 'likelihood' ? tr('运行概率评测', 'Run probability evaluation') : tr('运行答案评测', 'Run answer evaluation')}</button></div>
  </form>;
}
