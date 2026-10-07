import { useEffect, useRef, useState } from 'react';
import { ArrowRightIcon, CheckIcon, DownloadSimpleIcon, FolderOpenIcon, UploadSimpleIcon } from '@phosphor-icons/react';
import { jobsApi } from '../../shared/api/resources/jobs';
import { mediaApi } from '../../shared/api/resources/media';
import { quantizationApi, type QuantizationSource, type QuantizationWorkspace } from '../../shared/api/resources/quantization';
import { SourceLoadingControl } from './SourceLoadingControl';
import type { JobResource } from '../../shared/api/types';
import { ApiError } from '../../shared/api/client';
import { useJobStore } from '../../stores/jobStore';
import { useSettings } from '../settings/SettingsProvider';
import { WorkbenchFileDialog } from './WorkbenchFileDialog';
import './quantization-workbench.css';

type ArtifactKind = 'recipe' | 'imatrix';
type Picker = { title: string; directories: boolean; allowFiles?: boolean; initialPath: string; select: (path: string) => void };
const active = (job: JobResource) => ['queued', 'running', 'cancelling'].includes(job.status);
const join = (directory: string, name: string) => `${directory.replace(/\/+$/, '')}/${name}`;
const stamp = () => new Date().toISOString().replace(/[:.]/g, '-');
const message = (cause: unknown) => cause instanceof Error ? cause.message : String(cause);

export function QuantizationPage() {
  const { tr } = useSettings();
  const jobs = useJobStore((state) => state.jobs);
  const [workspace, setWorkspace] = useState<QuantizationWorkspace | null>(null);
  const [importDirectory, setImportDirectory] = useState('');
  const [exportDirectory, setExportDirectory] = useState('');
  const [sourcePath, setSourcePath] = useState('');
  const [source, setSource] = useState<QuantizationSource | null>(null);
  const [corpus, setCorpus] = useState('');
  const [applyChatTemplate, setApplyChatTemplate] = useState(true);
  const [layerwise, setLayerwise] = useState(true);
  const [imatrix, setImatrix] = useState('');
  const [recipe, setRecipe] = useState('');
  const [recipeJobId, setRecipeJobId] = useState<string | null>(null);
  const [targetBpw, setTargetBpw] = useState('4');
  const [candidates, setCandidates] = useState<string[]>([]);
  const [candidatesOpen, setCandidatesOpen] = useState(false);
  const [calibrated, setCalibrated] = useState(false);
  const [preserveService, setPreserveService] = useState(true);
  const [output, setOutput] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [recipeError, setRecipeError] = useState('');
  const [notice, setNotice] = useState('');
  const [picker, setPicker] = useState<Picker | null>(null);
  const [pendingImport, setPendingImport] = useState<{ id: string; kind: ArtifactKind } | null>(null);
  const [taskLogs, setTaskLogs] = useState<Record<string, string>>({});
  const upload = useRef<HTMLInputElement>(null);
  const uploadKind = useRef<ArtifactKind>('imatrix');
  const sourceVersion = useRef(0);
  const workJobs = jobs.filter((job) => job.kind.startsWith('quantization.'));

  useEffect(() => {
    let disposed = false;
    void Promise.all([quantizationApi.workspace(), jobsApi.jobs()]).then(([settings, records]) => {
      if (disposed) return;
      setWorkspace(settings); setImportDirectory(settings.import_directory); setExportDirectory(settings.export_directory); setCandidates(settings.candidates);
      records.forEach((job) => useJobStore.getState().addJob(job));
    }).catch((cause) => { if (!disposed) setError(message(cause)); });
    return () => { disposed = true; };
  }, []);

  useEffect(() => {
    if (!pendingImport) return;
    const job = jobs.find((item) => item.id === pendingImport.id);
    if (job?.status === 'succeeded' && typeof job.result?.output === 'string') {
      if (pendingImport.kind === 'recipe') { setRecipe(job.result.output); setRecipeJobId(job.id); }
      else setImatrix(job.result.output);
      setPendingImport(null);
    } else if (job && !active(job)) {
      setError(job.error?.message || tr('导入失败，请查看任务日志', 'Import failed; check the task log')); setPendingImport(null);
    }
  }, [jobs, pendingImport, tr]);

  async function run(action: () => Promise<unknown>, recipeAction = false) {
    setBusy(true); setError(''); setRecipeError(''); setNotice('');
    try { await action(); } catch (cause) { (recipeAction ? setRecipeError : setError)(validationMessage(cause)); } finally { setBusy(false); }
  }

  function validationMessage(cause: unknown) {
    if (!(cause instanceof ApiError)) return message(cause);
    const value = (key: string) => typeof cause.details?.[key] === 'number' ? Number(cause.details[key]).toFixed(4) : '—';
    if (cause.code === 'quantization_empty_candidates') return tr('请至少选择一个与源模型兼容的候选。', 'Select at least one source-compatible candidate.');
    if (cause.code === 'quantization_invalid_target') return tr('目标 bpw 必须是大于 0、不超过 32 的有限数值。', 'Target bpw must be finite and in (0, 32].');
    if (cause.code === 'quantization_no_feasible_budget') return tr(`所选候选的最低预算 ${value('minimum_bpw')} bpw 超出可用上限，当前没有可行配方。请更换候选，而不是继续调整目标。`, `The selected candidate minimum budget ${value('minimum_bpw')} bpw exceeds the available ceiling. No recipe is feasible; change the candidate set instead of adjusting the target.`);
    if (cause.code === 'quantization_target_above_source') return tr(`目标 ${value('target_bpw')} bpw 高于源模型实际平均 ${value('source_bpw')} bpw，请降低目标。`, `Target ${value('target_bpw')} bpw exceeds the source model average ${value('source_bpw')} bpw. Lower the target.`);
    if (cause.code === 'quantization_target_below_candidates') return tr(`目标 ${value('target_bpw')} bpw 低于所选候选的最低预算 ${value('minimum_bpw')} bpw，请提高目标或加入更低精度候选。`, `Target ${value('target_bpw')} bpw is below the selected candidate minimum budget ${value('minimum_bpw')} bpw. Raise the target or add lower-precision candidates.`);
    if (cause.code === 'quantization_target_above_candidates') return tr(`目标 ${value('target_bpw')} bpw 高于所选候选最高精度方案的 ${value('maximum_bpw')} bpw，请降低目标或加入更高精度候选。`, `Target ${value('target_bpw')} bpw exceeds the highest-precision candidate plan ${value('maximum_bpw')} bpw. Lower the target or add higher-precision candidates.`);
    return message(cause);
  }

  function chatRenderingMessage(job: JobResource) {
    const value = job.result?.chat_rendering as Record<string, unknown> | undefined;
    if (!value) return null;
    if (value.enabled === false) return tr('已保留原始语料格式，未自动添加聊天模板。', 'Original corpus format preserved; automatic chat templates were disabled.');
    if (value.status === 'already_prepared') return tr('语料已带聊天模板，未重复添加。', 'Corpus already had a chat template; no duplicate wrapping.');
    if (value.model_has_template === false) return tr('源模型未提供聊天模板，已保留原始语料格式。', 'Source model has no chat template; original corpus format preserved.');
    if (typeof value.wrapped === 'number' && typeof value.preformatted === 'number') {
      const count = value.wrapped + (typeof value.conversations === 'number' ? value.conversations : 0);
      return tr(`已套用模板 ${count} 条 · 保留已格式化语料 ${value.preformatted} 条`, `Template applied to ${count} records · ${value.preformatted} preformatted records preserved`);
    }
    return null;
  }

  function changeSource(path: string) {
    sourceVersion.current++; setSourcePath(path); setSource(null);
  }

  async function inspectSource() {
    const version = sourceVersion.current;
    const result = await quantizationApi.source(sourcePath);
    if (version === sourceVersion.current) {
      setSource(result);
      setOutput(join(join(exportDirectory, 'models'), `${result.path.split('/').filter(Boolean).at(-1)?.replace(/\.mfq$/i, '') || 'model'}-quantized.mfq`));
    }
  }

  async function submit(kind: string, payload: Record<string, unknown>) {
    const job = await jobsApi.createJob(`quantization.${kind}`, payload);
    useJobStore.getState().addJob(job);
    setNotice(tr('任务已注册，可在下方查看进度', 'Task registered. Track progress below.'));
    return job;
  }

  async function importArtifact(kind: ArtifactKind, path?: string, file?: File) {
    const payload: Record<string, unknown> = { kind };
    if (file) {
      const media = await mediaApi.uploadMedia(file);
      payload.media_id = media.media.id;
      payload.output = join(kind === 'imatrix' ? join(exportDirectory, 'imatrix') : exportDirectory, `${stamp()}-${kind}${kind === 'recipe' ? '.json' : '.imatrix'}`);
    } else payload.source = path;
    const job = await submit('import', payload);
    setPendingImport({ id: job.id, kind });
  }

  function browse(title: string, select: (path: string) => void, directories = false, allowFiles = false, initialPath = importDirectory) {
    setPicker({ title, select, directories, allowFiles, initialPath });
  }

  function artifactButtons(kind: ArtifactKind) {
    return <div className="qw-actions">
      <button type="button" disabled={busy || !workspace} onClick={() => browse(tr('选择导入文件', 'Select import file'), (path) => { void run(() => importArtifact(kind, path)); })}><FolderOpenIcon size={16} />{tr('从目录导入', 'Import from directory')}</button>
      <button type="button" disabled={busy || !workspace} onClick={() => { uploadKind.current = kind; upload.current?.click(); }}><UploadSimpleIcon size={16} />{tr('上传文件', 'Upload file')}</button>
    </div>;
  }

  const selectedRecipe = jobs.find((job) => job.id === recipeJobId);
  const groups = workspace?.candidate_groups || Object.fromEntries(['NVQ', 'NINT', 'MXFP4-SQ', 'MXFP8-SQ', 'FP8-SQ'].map((group) => [group, workspace?.candidates.filter((name) => name.startsWith(group)) || []]));
  const eligibleCandidates = workspace?.candidates.filter((name) => source?.eligible_candidates ? source.eligible_candidates.includes(name) : !name.includes('-SQ-')) || [];
  const selectedCandidates = candidates.filter((name) => eligibleCandidates.includes(name));
  const validTarget = Number.isFinite(Number(targetBpw)) && Number(targetBpw) > 0 && Number(targetBpw) <= 32;
  const policy = preserveService ? 'preserve' : 'allow';
  const title = (number: string, zh: string, en: string, detail?: string) => <header className="qw-section-title"><span>{number}</span><div><h3>{tr(zh, en)}</h3>{detail && <p>{detail}</p>}</div></header>;
  return <section className="quantization-workbench" aria-label={tr('量化工作台', 'Quantization workspace')}>
    <header className="qw-heading"><div><h2>{tr('量化工作台', 'Quantization workspace')}</h2><p>{tr('原始模型 → 精度配方 → MFQ 整模', 'Original model → precision recipe → complete MFQ model')}</p></div></header>
    <details className="qw-directories">
      <summary>{tr('默认目录', 'Default directories')}<span>{tr('导入 / 导出', 'Import / export')}</span></summary>
      <div className="qw-grid">
        <label><span>{tr('默认导入目录', 'Default import directory')}</span><div className="qw-path-row"><input value={importDirectory} onChange={(event) => setImportDirectory(event.target.value)} /><button type="button" disabled={!workspace} onClick={() => browse(tr('默认导入目录', 'Default import directory'), setImportDirectory, true)} aria-label={tr('浏览默认导入目录', 'Browse default import directory')}><FolderOpenIcon size={17} /></button></div></label>
        <label><span>{tr('默认导出目录', 'Default export directory')}</span><div className="qw-path-row"><input value={exportDirectory} onChange={(event) => setExportDirectory(event.target.value)} /><button type="button" disabled={!workspace} onClick={() => browse(tr('默认导出目录', 'Default export directory'), setExportDirectory, true, false, exportDirectory)} aria-label={tr('浏览默认导出目录', 'Browse default export directory')}><FolderOpenIcon size={17} /></button></div></label>
      </div>
      <div className="qw-actions"><button type="button" disabled={busy || !importDirectory.trim() || !exportDirectory.trim()} onClick={() => { void run(async () => {
        const settings = await quantizationApi.configure(importDirectory, exportDirectory);
        if (workspace && output.startsWith(workspace.export_directory + '/')) setOutput(join(settings.export_directory, output.slice(workspace.export_directory.length + 1)));
        setWorkspace(settings); setImportDirectory(settings.import_directory); setExportDirectory(settings.export_directory);
        setNotice(tr('默认目录已保存', 'Default directories saved'));
      }); }}>{tr('保存默认目录', 'Save default directories')}</button><small>{tr('配方直接导出；imatrix → imatrix/，整模 → models/。缺少的子目录自动创建。', 'Recipes export here; imatrix → imatrix/, models → models/. Missing subdirectories are created automatically.')}</small></div>
    </details>
    <div className="qw-section">
      {title('01', '选择原始模型', 'Select original model')}
      <label><span>{tr('源模型目录或 MFQ 文件', 'Source model directory or MFQ file')}</span><div className="qw-path-row"><input aria-label={tr('源模型目录或 MFQ 文件', 'Source model directory or MFQ file')} value={sourcePath} placeholder={importDirectory} onChange={(event) => changeSource(event.target.value)} /><button type="button" disabled={!workspace || busy} onClick={() => browse(tr('选择原始模型', 'Select original model'), changeSource, true, true)}><FolderOpenIcon size={16} />{tr('浏览', 'Browse')}</button><button type="button" disabled={busy || !sourcePath.trim()} onClick={() => { void run(inspectSource); }}>{tr('检查模型', 'Inspect model')}</button></div></label>
      <p className="qw-hint">{tr('支持 BF16 / FP16 / FP32，以及 FP8、MXFP8、MXFP4 等原生低精度 / QAT 权重。不支持 NVFP4 或已采用 MFQ 量化编码的权重。', 'Supports BF16 / FP16 / FP32 and native low-precision / QAT weights including FP8, MXFP8 and MXFP4. NVFP4 and MFQ-quantized weights are excluded.')}</p>
      {source && <div className="qw-source"><CheckIcon size={17} /><strong>{source.architecture}</strong><span>{(source.parameters / 1e9).toFixed(3)}B</span><span>{source.tensors} {tr('张量', 'tensors')}</span><span>{source.source_precisions?.join(' · ') || tr('原始权重', 'Original weights')}</span></div>}
    </div>
    <div className="qw-section">
      {title('02', '生成、下载或导入 imatrix', 'Generate, download or import imatrix', tr('DF-V1-AlphaQ 无需 imatrix；外部校准配方可在拟合时使用。', 'DF-V1-AlphaQ needs no imatrix. Calibrated external recipes can use one during fitting.'))}
      <div className="qw-actions">{artifactButtons('imatrix')}<button type="button" disabled={!workspace?.official_imatrix_url} onClick={() => { if (workspace?.official_imatrix_url) window.open(workspace.official_imatrix_url, '_blank', 'noopener,noreferrer'); }}><DownloadSimpleIcon size={16} />{tr('官方 imatrix · 待提供', 'Official imatrix · coming later')}</button></div>
      <details className="qw-subsection"><summary>{tr('从原始模型生成', 'Generate from original model')}</summary>
        <label><span>{tr('校准语料', 'Calibration corpus')}</span><div className="qw-path-row"><input value={corpus} onChange={(event) => setCorpus(event.target.value)} placeholder=".txt / .json / .jsonl" /><button type="button" disabled={!workspace} onClick={() => browse(tr('选择校准语料', 'Select calibration corpus'), setCorpus, true, true)}><FolderOpenIcon size={16} /></button></div></label>
        <div className="qw-service-choice"><label><input type="checkbox" checked={applyChatTemplate} onChange={(event) => setApplyChatTemplate(event.target.checked)} /><span>{tr('自动添加模型聊天模板', 'Automatically apply model chat template')}</span></label></div>
        <p className="qw-hint">{applyChatTemplate ? tr('未带聊天模板的语料将自动套用源模型的聊天模板；已格式化的语料不会重复添加。模型未提供聊天模板时保留原始格式。', 'Unformatted records use the source model’s chat template; already formatted records are not wrapped again. If the model has no chat template, the original format is preserved.') : tr('已关闭自动添加聊天模板，语料按原始格式处理。', 'Automatic chat templates are disabled; the original corpus format is preserved.')}</p>
        <SourceLoadingControl model={source?.format === 'hf' && source.imatrix_supported ? source.path : ''} purpose="imatrix" contextSize={16384} layerwise={layerwise} onChange={setLayerwise} />
        <div className="qw-actions"><button type="button" disabled={busy || source?.format !== 'hf' || !source.imatrix_supported || !corpus.trim()} onClick={() => { void run(async () => {
          const job = await submit('imatrix', { model: source!.path, corpus, apply_chat_template: applyChatTemplate, layerwise, output: join(join(exportDirectory, 'imatrix'), `imatrix-${stamp()}.imatrix`) });
          setPendingImport({ id: job.id, kind: 'imatrix' });
        }); }}>{tr('生成 imatrix', 'Generate imatrix')}</button><small>{tr('校准器目前支持 Qwen3.5 / Gemma4，包含这些架构的原生低精度 / QAT 权重；其他架构可导入 imatrix 或使用无校准配方。', 'The collector currently supports Qwen3.5 / Gemma4, including their native low-precision / QAT weights. Other architectures can import imatrix files or use uncalibrated recipes.')}</small></div>
      </details>
      {imatrix && <p className="qw-artifact-path">imatrix <span>{imatrix}</span></p>}
    </div>
    <div className="qw-section">
      {title('03', '生成或导入配方', 'Generate or import recipe')}
      <div className="qw-choice" role="group" aria-label={tr('配方类别', 'Recipe category')}><button type="button" aria-pressed={!calibrated} onClick={() => setCalibrated(false)}>{tr('无校准', 'Uncalibrated')}</button><button type="button" aria-pressed={calibrated} onClick={() => setCalibrated(true)}>{tr('有校准', 'Calibrated')}</button></div>
      {calibrated ? <p className="qw-placeholder">{tr('有校准的配方分配方法待确定。现有合法校准配方可从下方导入。', 'Calibrated allocation is not selected yet. Import an existing valid calibrated recipe below.')}</p> : <div className="qw-grid qw-recipe-controls">
        <label><span>{tr('分配方法', 'Allocation method')}</span><select value="DF-V1-AlphaQ" onChange={() => undefined}><option>DF-V1-AlphaQ</option></select></label>
        <label><span>{tr('目标 bpw', 'Target bpw')}</span><input type="number" min="0.01" max="32" step="0.1" value={targetBpw} onChange={(event) => setTargetBpw(event.target.value)} /></label>
        <div className="qw-candidates"><button type="button" className="qw-candidate-toggle" aria-expanded={candidatesOpen} aria-controls="qw-candidate-panel" onClick={() => setCandidatesOpen((open) => !open)}>{tr('候选集合', 'Candidates')}<span>{selectedCandidates.length} / {eligibleCandidates.length}</span></button><div id="qw-candidate-panel" hidden={!candidatesOpen} className={`qw-candidate-panel${Object.entries(groups).some(([group, names]) => group.includes('SQ') && names.length) ? ' qw-native-candidates' : ''}`}><div className="qw-candidate-toolbar"><button type="button" onClick={() => setCandidates(workspace?.candidates || [])}>{tr('全选', 'Select all')}</button><button type="button" onClick={() => setCandidates([])}>{tr('清空', 'Clear')}</button></div>{Object.entries(groups).filter(([, names]) => names.length > 0).map(([group, names]) => <fieldset key={group} className="qw-candidate-group" data-family={group}><legend>{group}</legend>{group.includes('SQ') && <p>{tr('仅对应原生权重可选，直接拟合原始码字与 scale。', 'Requires matching native weights; fit original codes and scales directly.')}</p>}<div className="qw-candidate-grid">{names.map((name) => <label key={name}><input type="checkbox" disabled={!eligibleCandidates.includes(name)} checked={selectedCandidates.includes(name)} onChange={(event) => setCandidates((current) => event.target.checked ? [...current.filter((item) => item !== name), name] : current.filter((item) => item !== name))} /><span>{name}</span></label>)}</div></fieldset>)}</div></div>
      </div>}
      <div className="qw-actions">{!calibrated && <button type="button" className="qw-primary" disabled={busy || !source} onClick={() => { void run(async () => {
        if (!selectedCandidates.length) throw new Error(tr('请至少选择一个与源模型兼容的候选。', 'Select at least one source-compatible candidate.'));
        if (!validTarget) throw new Error(tr('目标 bpw 必须是大于 0、不超过 32 的有限数值。', 'Target bpw must be finite and in (0, 32].'));
        const job = await submit('recipe', { input: source!.path, output: join(exportDirectory, `recipe-${stamp()}.json`), target_bpw: Number(targetBpw), candidates: selectedCandidates, method: 'DF-V1-AlphaQ', service_policy: policy });
        setPendingImport({ id: job.id, kind: 'recipe' });
      }, true); }}>{tr('生成配方', 'Generate recipe')}<ArrowRightIcon size={16} /></button>}{artifactButtons('recipe')}</div>
      {recipeError && <p className="qw-error" role="alert">{recipeError}</p>}
      {recipe && <div className="qw-recipe-result"><div><strong>{selectedRecipe?.result?.method as string || tr('已导入配方', 'Imported recipe')}</strong>{typeof selectedRecipe?.result?.estimated_bpw === 'number' && <span>{selectedRecipe.result.estimated_bpw.toFixed(3)} bpw</span>}<p>{recipe}</p></div><div className="qw-actions"><button type="button" disabled={busy || !recipeJobId} onClick={() => { void run(() => quantizationApi.exportRecipe(recipeJobId!)); }}><DownloadSimpleIcon size={16} />{tr('导出配方', 'Export recipe')}</button><button type="button" onClick={() => document.getElementById('qw-fit')?.scrollIntoView({ behavior: 'smooth', block: 'start' })}>{tr('导入量化', 'Use for quantization')}<ArrowRightIcon size={16} /></button></div></div>}
    </div>
    <div className="qw-section" id="qw-fit">
      {title('04', '量化并交付整模', 'Quantize and deliver complete model')}
      <label><span>{tr('量化配方', 'Quantization recipe')}</span><input readOnly value={recipe} placeholder={tr('从上方生成或导入合法配方', 'Generate or import a valid recipe above')} /></label>
      <label><span>{tr('整模输出路径', 'Complete model output path')}</span><div className="qw-path-row"><input value={output} placeholder={join(join(exportDirectory, 'models'), 'model.mfq')} onChange={(event) => setOutput(event.target.value)} /><button type="button" disabled={!workspace} onClick={() => browse(tr('整模输出目录', 'Model output directory'), (path) => setOutput(join(path, output.split('/').at(-1) || 'model.mfq')), true, false, join(exportDirectory, 'models'))}><FolderOpenIcon size={16} /></button></div></label>
      <div className="qw-service-choice" role="group" aria-label={tr('服务策略', 'Service policy')}><label><input type="radio" name="qw-policy" checked={preserveService} onChange={() => setPreserveService(true)} /><span>{tr('不影响服务', 'Preserve service')}</span></label><label><input type="radio" name="qw-policy" checked={!preserveService} onChange={() => setPreserveService(false)} /><span>{tr('影响服务', 'Allow service impact')}</span></label></div>
      <p className="qw-policy-note">{preserveService ? tr('优先保障服务：仅在低谷或空闲时拟合，并随资源压力降低批次。服务频繁或资源吃紧时，量化可能非常慢。正在执行的小批次不能中途抢占。', 'Prioritize service: fit only during quiet or idle periods and reduce batches under resource pressure. Frequent traffic or tight resources can make quantization very slow. A running small batch cannot be preempted.') : tr('持续拟合以加快完成，可能降低同时进行的推理吞吐并增加响应延迟。', 'Fit continuously for faster completion; concurrent inference may slow down and response latency can increase.')}</p>
      <div className="qw-actions"><button type="button" className="qw-primary" disabled={busy || !source || !recipe || !output.toLowerCase().endsWith('.mfq')} onClick={() => { void run(() => submit('fit', { input: source!.path, recipe, output, imatrix: selectedRecipe?.result?.method === 'DF-V1-AlphaQ' ? null : imatrix || null, service_policy: policy })); }}>{tr('注册量化任务', 'Register quantization task')}<ArrowRightIcon size={16} /></button><small>{tr('按张量分批拟合，完成后自动加入本地模型。已有文件不会被覆盖。', 'Fit tensors in batches, then register the complete local model. Existing files are never overwritten.')}</small></div>
    </div>
    <div className="qw-section qw-tasks">
      <header className="qw-task-heading"><h3>{tr('任务进度', 'Task progress')}</h3><span>{workJobs.filter(active).length} {tr('进行中', 'active')}</span></header>
      {!workJobs.length && <p className="qw-hint">{tr('尚无量化任务', 'No quantization tasks yet')}</p>}
      {workJobs.map((job) => <article key={job.id} className="qw-task"><div className="qw-task-heading"><strong>{({ 'quantization.recipe': tr('配方生成', 'Recipe generation'), 'quantization.fit': tr('整模量化', 'Model quantization'), 'quantization.imatrix': tr('imatrix 生成', 'imatrix generation'), 'quantization.import': tr('文件导入', 'Artifact import') })[job.kind] || job.kind}</strong><span>{({ queued: tr('排队中', 'Queued'), running: tr('进行中', 'Running'), cancelling: tr('取消中', 'Cancelling'), succeeded: tr('已完成', 'Completed'), failed: tr('失败', 'Failed'), cancelled: tr('已取消', 'Cancelled'), interrupted: tr('已中断', 'Interrupted') })[job.status]} · {(job.progress * 100).toFixed(0)}%</span></div>
        <progress max={1} value={job.progress} aria-label={tr('任务进度', 'Task progress')} />
        {job.progress_data?.phase === 'waiting' && <p className="qw-hint">{tr('等待服务低谷或可用资源', 'Waiting for quiet service or available resources')}</p>}
        {active(job) && job.progress_data?.source_loading?.fallback_reason === 'insufficient_memory' && <p className="qw-hint" role="status">{tr('空余内存不足，正在逐层生成。', 'Insufficient free memory; generating layerwise.')}</p>}
        {typeof job.progress_data?.tensor === 'string' && <p className="qw-artifact-path">{job.progress_data.tensor}</p>}
        {typeof job.result?.output === 'string' && <p className="qw-artifact-path">{job.result.output}</p>}
        {chatRenderingMessage(job) && <p className="qw-hint">{chatRenderingMessage(job)}</p>}
        {!!job.result?.loading && typeof job.result.loading === 'object' && 'fallback_reason' in job.result.loading && job.result.loading.fallback_reason === 'insufficient_memory' && <p className="qw-hint">{tr('启动时空余内存不足，已自动改为逐层生成。', 'Free memory was insufficient at startup; generation automatically used layerwise loading.')}</p>}
        {job.error && <p className="qw-error">{job.error.message}</p>}
        <div className="qw-actions">{active(job) ? <button type="button" disabled={busy || job.status === 'cancelling'} onClick={() => { void run(async () => { useJobStore.getState().addJob(await jobsApi.cancelJob(job.id)); }); }}>{tr('取消任务', 'Cancel task')}</button> : job.status === 'succeeded' && ['recipe', 'imatrix'].includes(String(job.result?.artifact_kind)) && <button type="button" onClick={() => {
          if (job.result?.artifact_kind === 'recipe') { setRecipe(String(job.result.output)); setRecipeJobId(job.id); } else setImatrix(String(job.result?.output));
        }}>{tr('使用此产物', 'Use this artifact')}</button>}<button type="button" disabled={busy} onClick={() => { void run(async () => { const logs = await jobsApi.jobEvents(job.id); setTaskLogs((current) => ({ ...current, [job.id]: logs.map((line) => line.message).join('\n') || tr('暂无日志', 'No log entries') })); }); }}>{tr('查看日志', 'View log')}</button></div>
        {taskLogs[job.id] && <pre className="qw-task-log">{taskLogs[job.id]}</pre>}
      </article>)}
    </div>
    {error && <p className="qw-feedback qw-error" role="alert">{error}</p>}
    {notice && <p className="qw-feedback" role="status">{notice}</p>}
    <input ref={upload} type="file" hidden accept=".json,.imatrix,.npz,.dat" onChange={(event) => { const file = event.target.files?.[0]; event.target.value = ''; if (file) void run(() => importArtifact(uploadKind.current, undefined, file)); }} />
    {picker && <WorkbenchFileDialog {...picker} onSelect={(path) => { picker.select(path); setPicker(null); }} onClose={() => setPicker(null)} />}
  </section>;
}
