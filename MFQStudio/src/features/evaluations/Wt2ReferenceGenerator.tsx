import { useEffect, useRef, useState } from 'react';
import { FolderOpenIcon } from '@phosphor-icons/react';
import { quantizationApi, type QuantizationSource } from '../../shared/api/resources/quantization';
import { WorkbenchFileDialog } from '../jobs/WorkbenchFileDialog';
import { SourceLoadingControl } from '../jobs/SourceLoadingControl';
import { useSettings } from '../settings/SettingsProvider';
import { errorMessage } from '../../app/formatters';
import '../jobs/quantization-workbench.css';

export interface GeneratedWt2Reference {
  id: string; output: string; manifest: string; context_size: number; chunks: number; parallel: number; dataset_id: string;
}

export function Wt2ReferenceGenerator({ dataset, contextSize, chunks, available, busy, outputRoot, submit }: {
  dataset: string; contextSize: number; chunks: number; available: boolean; busy: boolean; outputRoot: string;
  submit: (kind: string, payload: Record<string, unknown>) => Promise<void>;
}) {
  const { tr } = useSettings();
  const [path, setPath] = useState('');
  const [source, setSource] = useState<QuantizationSource | null>(null);
  const [layerwise, setLayerwise] = useState(true);
  const [output, setOutput] = useState('');
  const [picker, setPicker] = useState<string | null>(null);
  const [checking, setChecking] = useState(false);
  const [error, setError] = useState('');
  const version = useRef(0);
  useEffect(() => () => { version.current++; }, []);
  function select(value: string) { version.current++; setPath(value); setSource(null); setError(''); setChecking(false); }
  async function inspect() {
    const current = ++version.current;
    setChecking(true); setError('');
    try { const next = await quantizationApi.source(path); if (current === version.current) setSource(next); }
    catch (cause) { if (current === version.current) setError(errorMessage(cause)); }
    finally { if (current === version.current) setChecking(false); }
  }
  const supported = source?.format === 'hf' && source.imatrix_supported;
  return <details className="evaluation-advanced wt2-reference-generator">
    <summary>{tr('从原始模型生成 WT2 logits', 'Generate WT2 logits from original model')}</summary>
    <div className="evaluation-fields evaluation-fields-pair">
      <label className="evaluation-wide">{tr('原始模型目录', 'Original model directory')}<div className="qw-path-row"><input aria-label={tr('原始模型目录', 'Original model directory')} value={path} onChange={(event) => select(event.target.value)} placeholder="HF safetensors" /><button type="button" aria-label={tr('浏览原始模型', 'Browse original model')} disabled={checking} onClick={() => { void quantizationApi.workspace().then((workspace) => setPicker(workspace.import_directory)).catch((cause) => setError(errorMessage(cause))); }}><FolderOpenIcon size={16} /></button><button type="button" disabled={!path.trim() || checking} onClick={() => { void inspect(); }}>{tr('检查模型', 'Inspect model')}</button></div></label>
      <label className="evaluation-wide">{tr('参考 logits 输出路径', 'Reference logits output path')}<input value={output} onChange={(event) => setOutput(event.target.value)} placeholder={`${outputRoot}/evaluations/references/wt2.logits`} /></label>
    </div>
    {source && <p className="evaluation-note">{source.architecture} · {source.source_precisions?.join(' / ') || 'HF'} · {(source.parameters / 1e9).toFixed(2)}B</p>}
    <SourceLoadingControl model={supported ? source!.path : ''} purpose="wt2" contextSize={contextSize} layerwise={layerwise} onChange={setLayerwise} />
    <p className="evaluation-note">{tr('沿用上方官方 WT2 集合和下方 ctx / 窗口数，生成 logits 与 manifest。固定原始文本，不添加聊天模板；并行序列为 1。支持 Qwen3.5 / Gemma4 的原始及原生低精度 / QAT 权重。', 'Uses the official WT2 collection above and ctx / windows below, producing logits and a manifest. Raw text is unchanged, without chat templates; one parallel sequence. Supports original and native low-precision / QAT Qwen3.5 / Gemma4 weights.')}</p>
    {source && !supported && <p role="status" className="evaluation-note">{tr('此源模型架构尚未接入参考生成器。', 'This source architecture is not integrated with the reference generator yet.')}</p>}
    {error && <p role="alert" className="evaluation-note">{error}</p>}
    <button type="button" className="evaluation-run" disabled={busy || checking || !available || !supported || !dataset || !Number.isInteger(contextSize) || contextSize < 32 || !Number.isInteger(chunks) || chunks < 1} onClick={() => { void submit('reference.wikitext2', {
      model: source!.path, dataset_id: dataset, context_size: contextSize, chunks, layerwise,
      output: output.trim() || `${outputRoot}/evaluations/references/wt2-${Date.now()}.logits`,
    }); }}>{tr('生成 WT2 logits', 'Generate WT2 logits')}</button>
    {picker && <WorkbenchFileDialog initialPath={picker} directories title={tr('选择原始模型', 'Select original model')} onSelect={(value) => { select(value); setPicker(null); }} onClose={() => setPicker(null)} />}
  </details>;
}
