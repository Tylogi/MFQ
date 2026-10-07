import { useEffect, useMemo, useState } from 'react';
import { useSearchParams } from 'react-router';
import { Icon, EmptyPanel } from '../../app/display';
import { errorMessage, formatBytes } from '../../app/formatters';
import { useRuntime } from '../../app/RuntimeProvider';
import { modelsApi } from '../../shared/api/resources/models';
import type { CheckpointAnalysis, ModelArtifact } from '../../shared/api/types';
import { ModelVendorMark } from '../../app/ModelVendorMark';
import { useSettings } from '../settings/SettingsProvider';
import { ArchitectureDiagram } from './ArchitectureDiagram';
import { AdvancedDetails } from './AdvancedDetails';
import { KvCachePlanner } from '../models/KvCachePlanner';
import { ExpertHeatmap, LayerBudget, PrecisionLegend, ProjectionBars } from './PrecisionCharts';
import { categoryLabel, parameterLabel, precisionRange } from './analysisData';

export function AnalysisPage() {
  const { tr } = useSettings();
  const { selectedModel, connectionRevision } = useRuntime();
  const [search, setSearch] = useSearchParams();
  const [models, setModels] = useState<ModelArtifact[]>([]);
  const [selected, setSelected] = useState(search.get('checkpoint') ?? '');
  const [analysis, setAnalysis] = useState<CheckpointAnalysis | null>(null);
  const [loading, setLoading] = useState(true);
  const [listing, setListing] = useState(true);
  const [error, setError] = useState('');
  const [revision, setRevision] = useState(0);
  const [visualization, setVisualization] = useState<'experts' | 'projections'>('experts');
  const [category, setCategory] = useState('all');
  const [tensorSearch, setTensorSearch] = useState('');
  const [tensorPage, setTensorPage] = useState(0);
  useEffect(() => {
    let current = true;
    setListing(true); setError(''); setModels([]); setAnalysis(null);
    void modelsApi.modelArtifacts(revision > 0).then(items => {
      if (!current) return;
      setModels(items);
      setSelected(previous => items.some(item => item.id === previous) ? previous
        : items.find(item => item.name === selectedModel)?.id ?? items.find(item => item.complete)?.id ?? items[0]?.id ?? '');
    }).catch(cause => { if (current) setError(errorMessage(cause)); })
      .finally(() => { if (current) { setListing(false); setLoading(false); } });
    return () => { current = false; };
  }, [connectionRevision, revision]);
  useEffect(() => {
    if (listing || !selected) return;
    const controller = new AbortController();
    setLoading(true); setAnalysis(null); setError(''); setTensorPage(0); setTensorSearch(''); setCategory('all');
    const checkpoint = models.find(item => item.id === selected);
    if (checkpoint && !checkpoint.complete) {
      setError(tr('分片不完整，补全文件后才能分析模型。', 'Model shards are incomplete. Restore the missing files before analysis.'));
      setLoading(false);
      return;
    }
    void modelsApi.checkpointAnalysis(selected, controller.signal).then(result => {
      if (!controller.signal.aborted) setAnalysis(result);
    }).catch(cause => { if (!controller.signal.aborted) setError(errorMessage(cause)); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [selected, listing, connectionRevision, revision]);
  const range = useMemo(() => analysis ? precisionRange(analysis) : [0, 8] as [number, number], [analysis]);
  const categories = [...new Set(analysis?.projections.map(item => item.category) ?? [])];
  const projections = analysis?.projections.filter(item => category === 'all' || item.category === category) ?? [];
  const tensors = analysis?.tensors.filter(item => !tensorSearch || item.name.toLowerCase().includes(tensorSearch.toLowerCase())) ?? [];
  const pages = Math.max(1, Math.ceil(tensors.length / 30));
  const currentPage = Math.min(tensorPage, pages - 1);
  const selectedArtifact = models.find(item => item.id === selected);
  const pleParameters = analysis?.tensors.filter(item => item.category === 'ple').reduce((sum, item) => sum + item.parameters, 0) ?? 0;
  return <div className="analysis-page">
    <section className="analysis-checkpoint-bar">
      <label><span>{tr('模型', 'Model')}</span><select aria-label={tr('选择模型', 'Select model')} disabled={listing || !models.length}
        value={selected} onChange={event => { setSelected(event.target.value); setSearch({ checkpoint: event.target.value }, { replace: true }); }}>
        {!models.length && <option value="">{listing ? tr('正在读取…', 'Reading…') : tr('没有本地模型', 'No local models')}</option>}
        {models.map(item => <option key={item.id} value={item.id}>{item.name}{!item.complete ? ` · ${tr('分片不全', 'Incomplete shards')}` : ''}</option>)}</select></label>
      {selectedArtifact && <ModelVendorMark name={selectedArtifact.name} architecture={selectedArtifact.architecture} size={25} />}
      <button aria-label={tr('刷新模型', 'Refresh models')} disabled={listing} onClick={() => setRevision(value => value + 1)} type="button"><Icon name="refresh" size={15} /></button>
    </section>
    {loading || listing ? <div className="analysis-loading" role="status"><Icon name="search" size={22} /><strong>{tr('正在读取模型元数据', 'Reading model metadata')}</strong></div>
      : error ? <div className="analysis-error" role="alert"><Icon name="info" size={18} /><div><strong>{tr('暂时无法分析', 'Analysis unavailable')}</strong><p>{error}</p></div>
        <button onClick={() => setRevision(value => value + 1)} type="button">{tr('重试', 'Retry')}</button></div>
      : !analysis ? <EmptyPanel icon="search" title={tr('选择一个模型', 'Choose a model')} message={tr('已注册的本地模型都可以在这里查看。', 'Inspect registered local models here.')} />
      : <div className="analysis-layout">
        <ArchitectureDiagram analysis={analysis} tr={tr} />
        <div className="analysis-main">
          <section className="analysis-panel analysis-summary">
            <div className="analysis-panel-heading"><h2>{tr('模型结构', 'Model structure')}</h2><span>{analysis.format.toUpperCase()} · {selectedArtifact?.shard_count} {tr('分片', 'shards')}</span></div>
            <dl className="analysis-facts">
              <div className="analysis-fact-architecture"><dt>{tr('架构', 'Architecture')}</dt><dd>{analysis.architecture}</dd></div>
              <div><dt>{tr('层数 / 隐藏维度', 'Layers / hidden size')}</dt><dd>{analysis.layer_count} / {analysis.hidden_size?.toLocaleString() ?? '—'}</dd></div>
              <div><dt>{tr('注意力头 Q / KV', 'Attention heads Q / KV')}</dt><dd>{analysis.attention_heads ?? '—'} / {analysis.kv_heads ?? '—'}</dd></div>
              <div><dt>{tr('头维度', 'Head dimension')}</dt><dd>{analysis.head_dim ?? '—'}</dd></div>
              <div><dt>{tr('GDN 头 K / V', 'GDN heads K / V')}</dt><dd>{analysis.linear_key_heads ?? '—'} / {analysis.linear_value_heads ?? '—'}</dd></div>
              <div><dt>{tr('路由专家 / 激活', 'Routed / active experts')}</dt><dd>{analysis.expert_count || '—'} / {analysis.experts_per_token ?? '—'}</dd></div>
              <div><dt>{tr('参数量', 'Parameters')}</dt><dd>{parameterLabel(analysis.parameters)}</dd>{pleParameters > 0 && <small>{tr('含', 'Includes')} {parameterLabel(pleParameters)} PLE</small>}</div>
              <div><dt>{tr('平均 bpw', 'Average bpw')}</dt><dd>{analysis.average_bpw?.toFixed(3) ?? '—'}</dd></div>
              <div><dt>{tr('权重存储', 'Stored weights')}</dt><dd>{(analysis.stored_bytes / 2 ** 30).toFixed(1)} GiB</dd></div>
            </dl>
            <AdvancedDetails analysis={analysis} tr={tr} />
            {Object.keys(analysis.attention_distribution).some(kind => kind !== 'GDN') && <KvCachePlanner key={analysis.model_id} profile={analysis.cache_profile} tr={tr} />}
            <div className="analysis-attention-distribution"><div className="analysis-section-heading"><h3>{tr('注意力分布', 'Attention distribution')}</h3>
              <div>{Object.entries(analysis.attention_distribution).map(([kind, ids]) => <span key={kind}><i className={`attention-kind attention-${kind.toLowerCase()}`} />{kind} <b>{ids.length}</b></span>)}</div></div>
              <div className="attention-layer-strip">{analysis.layers.map(layer => <span key={layer.layer} className={`attention-kind attention-${layer.attention_type.toLowerCase()}`}
                title={`L${layer.layer} · ${layer.attention_type}`} aria-label={`L${layer.layer} · ${layer.attention_type}`} />)}</div>
              <div className="analysis-strip-labels"><span>L0</span><span>L{Math.max(0, analysis.layer_count - 1)}</span></div>
            </div>
            <div className="analysis-section-heading analysis-projection-heading"><h3>{tr('各类投影', 'Projection breakdown')}</h3>
              <select aria-label={tr('投影分类', 'Projection category')} value={category} onChange={event => setCategory(event.target.value)}>
                <option value="all">{tr('全部组件', 'All components')}</option>{categories.map(item => <option key={item} value={item}>{categoryLabel(item, tr)}</option>)}</select></div>
            <div className="analysis-table-scroll analysis-projection-table"><table><thead><tr>
              <th>{tr('组件 / 投影', 'Component / projection')}</th><th>{tr('参数量', 'Parameters')}</th><th>bpw</th><th>{tr('格式', 'Formats')}</th></tr></thead>
              <tbody>{projections.map((item, index) => <tr key={`${item.category}:${item.name}:${index}`}><td><span>{categoryLabel(item.category, tr)}</span><strong>{item.name}</strong></td>
                <td title={item.parameters.toLocaleString()}>{parameterLabel(item.parameters)}</td><td>{item.average_bpw?.toFixed(3) ?? '—'}</td><td>{item.formats.join(' · ')}</td></tr>)}</tbody></table></div>
          </section>
          <section className="analysis-panel analysis-visualizations">
            <div className="analysis-panel-heading"><h2>{tr('量化可视化', 'Quantization visualization')}</h2><Icon name="chart" size={16} /></div>
            <div className="analysis-tabs" role="tablist" aria-label={tr('量化图表', 'Quantization charts')}>
              <button role="tab" id="analysis-expert-tab" aria-controls="analysis-chart-content" aria-selected={visualization === 'experts'} onClick={() => setVisualization('experts')} type="button">{tr('路由专家', 'Routed experts')}</button>
              <button role="tab" id="analysis-projection-tab" aria-controls="analysis-chart-content" aria-selected={visualization === 'projections'} onClick={() => setVisualization('projections')} type="button">{tr('注意力与投影', 'Attention & projections')}</button></div>
            <div id="analysis-chart-content" role="tabpanel" aria-labelledby={visualization === 'experts' ? 'analysis-expert-tab' : 'analysis-projection-tab'}>
              {visualization === 'experts' ? analysis.experts.length ? <>
                <PrecisionLegend range={range} tr={tr} /><ExpertHeatmap analysis={analysis} range={range} tr={tr} />
                <LayerBudget analysis={analysis} tr={tr} />
                <div className="analysis-projection-maps">{['gate', 'up', 'down'].map(projection =>
                  <ExpertHeatmap key={projection} analysis={analysis} projection={projection} range={range} tr={tr} />)}</div>
              </> : <div className="analysis-empty-chart"><Icon name="chart" size={22} /><span>{tr('这个检查点没有可读取的路由专家精度，稠密与注意力投影见另一页。', 'No routed expert precision metadata. Inspect dense and attention projections in the other tab.')}</span>
                <LayerBudget analysis={analysis} tr={tr} /></div>
                : <><ProjectionBars analysis={analysis} tr={tr} /><LayerBudget analysis={analysis} tr={tr} /></>}
            </div>
            <p className="analysis-method">{tr('bpw = 实际存储位数 / 参数量，含量化元数据、码本与格式开销；平均值按参数量加权。不含资源文件、整数缓冲、KV Cache 或运行时展开。', 'bpw = stored bits / parameters, including quantization metadata, codebooks and format overhead; averages are parameter-weighted. Excludes assets, integer buffers, KV cache and runtime expansion.')}</p>
          </section>
          <details className="analysis-panel analysis-tensor-details"><summary>{tr('张量明细', 'Tensor details')}<span>{analysis.tensors.length}</span></summary>
            <div className="analysis-tensor-toolbar"><label><Icon name="search" size={14} /><input aria-label={tr('搜索张量', 'Search tensors')} placeholder={tr('按张量名称搜索', 'Search tensor names')}
              value={tensorSearch} onChange={event => { setTensorSearch(event.target.value); setTensorPage(0); }} /></label><span>{tensors.length} {tr('项', 'items')}</span></div>
            <div className="analysis-table-scroll"><table><thead><tr><th>{tr('张量 / 形状', 'Tensor / shape')}</th><th>{tr('参数量', 'Parameters')}</th><th>{tr('存储', 'Storage')}</th><th>bpw</th><th>{tr('格式', 'Format')}</th></tr></thead>
              <tbody>{tensors.slice(currentPage * 30, (currentPage + 1) * 30).map(item => <tr key={item.name}><td><strong>{item.name}</strong><span>{item.shape.join(' × ') || '—'}</span></td><td>{parameterLabel(item.parameters)}</td>
                <td>{formatBytes(item.stored_bytes)}</td><td>{item.bpw?.toFixed(3) ?? '—'}</td><td>{item.format}</td></tr>)}</tbody></table></div>
            <div className="analysis-pagination"><button disabled={currentPage === 0} onClick={() => setTensorPage(value => value - 1)} type="button">{tr('上一页', 'Previous')}</button>
              <span>{currentPage + 1} / {pages}</span><button disabled={currentPage + 1 >= pages} onClick={() => setTensorPage(value => value + 1)} type="button">{tr('下一页', 'Next')}</button></div>
          </details>
          {analysis.warnings.length > 0 && <details className="analysis-panel analysis-warnings"><summary>{tr('元数据提示', 'Metadata notices')}<span>{analysis.warnings.length}</span></summary>
            <ul>{analysis.warnings.map((warning, index) => <li key={index}>{warning}</li>)}</ul></details>}
        </div>
      </div>}
  </div>;
}
