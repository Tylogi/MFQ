import { useEffect, useMemo, useRef, useState } from 'react';
import type { KeyboardEvent, MouseEvent } from 'react';
import type { CheckpointAnalysis } from '../../shared/api/types';
import { averageBpw, categoryLabel, expertPoints, layerBudgetSeries, parameterLabel, precisionColor, projectionSeries, specialTensorBudgets, type Translate } from './analysisData';

export function PrecisionLegend({ range, tr }: { range: [number, number]; tr: Translate }) {
  return <div className="precision-legend"><span>{range[0].toFixed(1)}</span>
    <i style={{ background: `linear-gradient(90deg, ${precisionColor(range[0], ...range)}, ${precisionColor(range[1], ...range)})` }} />
    <span>{range[1].toFixed(1)} bpw</span><small>{tr('精度越高，颜色越深', 'Higher precision, darker color')}</small></div>;
}

export function ExpertHeatmap({ analysis, projection, range, tr }: {
  analysis: CheckpointAnalysis; projection?: string; range: [number, number]; tr: Translate;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const points = useMemo(() => expertPoints(analysis.experts, projection), [analysis.experts, projection]);
  const average = useMemo(() => averageBpw(points.values()), [points]);
  const [hover, setHover] = useState<{ layer: number; expert: number } | null>(null);
  const title = projection ? `${projection[0].toUpperCase()}${projection.slice(1)}` : tr('路由专家精度总览', 'Routed expert precision');
  const point = hover ? points.get(`${hover.layer}:${hover.expert}`) : null;
  const rowHeight = 7;
  useEffect(() => {
    const element = canvas.current;
    if (!element) return;
    const context = element.getContext('2d');
    if (!context) return;
    element.width = Math.max(1, analysis.expert_count * 2);
    element.height = Math.max(1, analysis.layer_count * rowHeight);
    context.fillStyle = '#e9e7e3';
    context.fillRect(0, 0, element.width, element.height);
    for (const item of points.values()) {
      context.fillStyle = precisionColor(item.bpw, ...range);
      context.fillRect(item.expert * 2, item.layer * rowHeight, 2, rowHeight);
    }
  }, [analysis.expert_count, analysis.layer_count, points, range]);
  function locate(event: MouseEvent<HTMLCanvasElement>) {
    const rect = event.currentTarget.getBoundingClientRect();
    setHover({ expert: Math.max(0, Math.min(analysis.expert_count - 1, Math.floor((event.clientX - rect.left) / rect.width * analysis.expert_count))),
      layer: Math.max(0, Math.min(analysis.layer_count - 1, Math.floor((event.clientY - rect.top) / rect.height * analysis.layer_count))) });
  }
  function keyboard(event: KeyboardEvent<HTMLCanvasElement>) {
    if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(event.key)) return;
    event.preventDefault();
    const current = hover ?? { expert: 0, layer: 0 };
    setHover({ expert: Math.max(0, Math.min(analysis.expert_count - 1, current.expert + (event.key === 'ArrowRight' ? 1 : event.key === 'ArrowLeft' ? -1 : 0))),
      layer: Math.max(0, Math.min(analysis.layer_count - 1, current.layer + (event.key === 'ArrowDown' ? 1 : event.key === 'ArrowUp' ? -1 : 0))) });
  }
  return <section className="precision-chart expert-heatmap">
    <div className="precision-chart-heading"><div className="precision-chart-title"><h3>{title}</h3>
      <span className="precision-chart-average">{tr('平均', 'Average')} {average?.toFixed(3) ?? '—'} bpw</span></div>
      <span>{tr('层 × 专家', 'Layer × expert')}</span></div>
    <div className="expert-map-layout">
      <div className="heatmap-y-axis"><span>{tr('层', 'Layer')}</span>{[...new Set([0, Math.floor((analysis.layer_count - 1) / 2), analysis.layer_count - 1])].map(layer =>
        <i key={layer} style={{ top: `${(layer + 0.5) / analysis.layer_count * 100}%` }}>{layer}</i>)}</div>
      <div className="heatmap-plot"><canvas ref={canvas} role="img" tabIndex={0} onMouseMove={locate} onClick={locate}
        onMouseLeave={() => setHover(null)} onFocus={() => setHover({ layer: 0, expert: 0 })} onBlur={() => setHover(null)} onKeyDown={keyboard}
        aria-label={`${title}: ${analysis.layer_count} ${tr('层', 'layers')}, ${analysis.expert_count} ${tr('专家；方向键查看精度', 'experts; arrow keys inspect precision')}`} />
        {hover && <div className="heatmap-crosshair" style={{ left: `${hover.expert / analysis.expert_count * 100}%`, top: `${hover.layer / analysis.layer_count * 100}%`,
          width: `${100 / analysis.expert_count}%`, height: `${100 / analysis.layer_count}%` }} />}
      </div>
      <div className="heatmap-x-axis"><span>0</span><span>{tr('专家号', 'Expert ID')}</span><span>{analysis.expert_count - 1}</span></div>
    </div>
    <div className="chart-readout" aria-live="polite">
      {hover ? <><strong>L{hover.layer} · E{hover.expert}</strong><span>{point ? `${point.bpw.toFixed(3)} bpw` : tr('未报告', 'Not reported')}</span>
        {point?.parts.map(part => <small key={part.projection}>{part.projection} {part.bpw.toFixed(3)} · {part.format}</small>)}</>
        : <span>{tr('悬停或点击查看专家；层号从 0 开始。', 'Hover or click to inspect an expert; layers are zero-based.')}</span>}
    </div>
  </section>;
}

export function LayerBudget({ analysis, tr }: { analysis: CheckpointAnalysis; tr: Translate }) {
  const [mode, setMode] = useState<'storage' | 'bpw'>('bpw');
  const [active, setActive] = useState<number | null>(null);
  const layers = useMemo(() => layerBudgetSeries(analysis), [analysis]);
  const special = useMemo(() => specialTensorBudgets(analysis), [analysis]);
  const series = [
    { key: 'moe' as const, label: 'MoE' }, { key: 'attention' as const, label: 'Attention' }, { key: 'total' as const, label: tr('总计', 'Total') },
  ].map(item => ({ ...item, values: layers.map(layer => mode === 'storage' ? layer[item.key].storedBytes / 2 ** 30 : layer[item.key].bpw) }));
  const maximum = Math.max(1, ...series.flatMap(item => item.values).filter((value): value is number => value != null)) * 1.12;
  const x = (index: number) => 52 + index / Math.max(1, analysis.layer_count - 1) * 588;
  const y = (value: number) => 172 - value / maximum * 142;
  const selected = active == null ? null : layers[active];
  const path = (values: (number | null)[]) => {
    const commands: string[] = [];
    let drawing = false;
    for (let i = 0; i < values.length; i++) {
      if (values[i] == null) { drawing = false; continue; }
      commands.push(`${drawing ? 'L' : 'M'}${x(i)} ${y(values[i]!)}`); drawing = true;
    }
    return commands.join(' ');
  };
  return <section className="precision-chart layer-budget">
    <div className="precision-chart-heading"><h3>{tr('层预算', 'Layer budget')}</h3>
      <div className="analysis-segmented"><button aria-pressed={mode === 'storage'} onClick={() => setMode('storage')} type="button">GiB</button>
        <button aria-pressed={mode === 'bpw'} onClick={() => setMode('bpw')} type="button">bpw</button></div></div>
    <div className="budget-legend">{series.map(item => <span key={item.key} className={`budget-series-${item.key}`}><i />{item.label}</span>)}</div>
    <svg viewBox="0 0 670 208" role="img" aria-label={tr('MoE、Attention 与总计的逐层预算折线图', 'Per-layer budget for MoE, Attention and Total')}>
      <text x="52" y="18">{mode === 'storage' ? 'GiB' : 'bpw'}</text>
      {[0, 0.5, 1].map(step => <g key={step}><line x1="52" x2="640" y1={y(step * maximum)} y2={y(step * maximum)} className="chart-gridline" />
        <text x="43" y={y(step * maximum) + 4} textAnchor="end">{(step * maximum).toFixed(1)}</text></g>)}
      {series.map(item => <g key={item.key} className={`budget-series-${item.key}`}>
        <path d={path(item.values)} className="budget-line" />
        {item.values.map((value, index) => value != null && <circle key={index} cx={x(index)} cy={y(value)} r={active === index ? 4 : 2} className="budget-dot" />)}
      </g>)}
      {layers.map((layer, index) => <g key={layer.layer}>
        <rect x={x(index) - 588 / Math.max(2, analysis.layer_count) / 2} y="24" width={588 / Math.max(1, analysis.layer_count - 1)} height="152" fill="transparent"
          role="button" tabIndex={0} aria-label={`L${layer.layer}: ${series.map(item => `${item.label} ${item.values[index]?.toFixed(3) ?? '—'}`).join(', ')} ${mode === 'storage' ? 'GiB' : 'bpw'}`}
          onMouseEnter={() => setActive(index)} onFocus={() => setActive(index)} onClick={() => setActive(index)}
          onKeyDown={event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); setActive(index); } }} />
      </g>)}
      <text x="52" y="197">0</text><text x="346" y="197" textAnchor="middle">{tr('层号', 'Layer ID')}</text><text x="640" y="197" textAnchor="end">{analysis.layer_count - 1}</text>
    </svg>
    <div className="chart-readout">{selected ? <><strong>L{selected.layer} · {selected.attention_type}</strong>
      {series.map(item => <span key={item.key} className={`budget-readout-series budget-series-${item.key}`}><i />{item.label}
        <span>{(selected[item.key].storedBytes / 2 ** 30).toFixed(3)} GiB · {selected[item.key].bpw?.toFixed(3) ?? '—'} bpw</span></span>)}
      </> : <span>{tr('悬停或点击查看每层的 MoE、Attention 和总预算。', 'Hover or click to inspect MoE, Attention and the total budget of each layer.')}</span>}</div>
    <p className="budget-note">{tr('MoE 含共享专家与路由器；Attention 含 GDN；总计包含其他层权重。所有曲线均不含 PLE 与 KV Cache。', 'MoE includes shared experts and routers; Attention includes GDN; Total includes other layer weights. All curves exclude PLE and KV cache.')}</p>
    {special.length > 0 && <section className="analysis-special-weights">
      <div className="analysis-section-heading"><h3>{tr('特殊张量', 'Special tensors')}</h3><span>{tr('不计入层预算', 'Excluded from layer budgets')}</span></div>
      <div className="analysis-table-scroll"><table><thead><tr><th>{tr('类型', 'Type')}</th><th>{tr('参数量', 'Parameters')}</th><th>{tr('存储', 'Storage')}</th><th>{tr('平均 bpw', 'Average bpw')}</th></tr></thead>
        <tbody>{special.map(item => <tr key={item.category}><td>{categoryLabel(item.category, tr)}</td><td>{parameterLabel(item.parameters)}</td>
          <td>{(item.storedBytes / 2 ** 30).toFixed(3)} GiB</td><td>{item.bpw?.toFixed(3) ?? '—'}</td></tr>)}</tbody></table></div>
    </section>}
  </section>;
}

export function ProjectionBars({ analysis, tr }: { analysis: CheckpointAnalysis; tr: Translate }) {
  const options = analysis.projections.filter(projection => !['routed_experts', 'ple'].includes(projection.category) &&
    analysis.tensors.some(tensor => tensor.category === projection.category && tensor.projection === projection.name && tensor.layer != null && tensor.shape.length >= 2));
  const [choice, setChoice] = useState('');
  const [active, setActive] = useState<number | null>(null);
  const option = options.find(item => `${item.category}:${item.name}` === choice) ?? options.find(item => item.category === 'attention' && item.name === 'query') ?? options[0];
  const series = useMemo(() => option ? projectionSeries(analysis.tensors, option.category, option.name, analysis.layer_count) : [], [analysis.tensors, analysis.layer_count, option]);
  const average = useMemo(() => averageBpw(series), [series]);
  if (!option) return null;
  const maximum = Math.max(1, ...series.map(item => item.bpw ?? 0)) * 1.1;
  const width = 588 / Math.max(1, analysis.layer_count);
  const selected = active == null ? null : series[active];
  return <section className="precision-chart projection-bars">
    <div className="precision-chart-heading"><div className="precision-chart-title"><h3>{tr('非专家投影精度', 'Non-expert projection precision')}</h3>
      <span className="precision-chart-average">{tr('平均', 'Average')} {average?.toFixed(3) ?? '—'} bpw</span></div>
      <select aria-label={tr('选择投影', 'Select projection')} value={`${option.category}:${option.name}`} onChange={event => { setChoice(event.target.value); setActive(null); }}>
        {options.map(item => <option key={`${item.category}:${item.name}`} value={`${item.category}:${item.name}`}>{item.category === 'attention' && analysis.attention_distribution.QSA ? 'QSA' : item.category.toUpperCase()} · {item.name}</option>)}</select></div>
    <svg viewBox="0 0 670 214" role="img" aria-label={`${option.name} ${tr('逐层 bpw 柱状热力图', 'per-layer bpw bar heatmap')}`}>
      {[0, 0.5, 1].map(step => <g key={step}><line x1="52" x2="640" y1={172 - step * 142} y2={172 - step * 142} className="chart-gridline" />
        <text x="43" y={176 - step * 142} textAnchor="end">{(step * maximum).toFixed(1)}</text></g>)}
      <text x="52" y="18">bpw</text>
      {series.map((item, index) => <g key={item.layer}>
        {item.bpw != null && <rect x={52 + index * width + 1} y={172 - item.bpw / maximum * 142} width={Math.max(1, width - 2)} height={item.bpw / maximum * 142}
          fill={precisionColor(item.bpw, 0, maximum / 1.1)} rx="1" />}
        <rect x={52 + index * width} y="24" width={width} height="152" fill="transparent" tabIndex={0} role="button"
          aria-label={`L${index}: ${item.bpw?.toFixed(3) ?? tr('无此投影', 'No projection')} bpw`}
          onMouseEnter={() => setActive(index)} onFocus={() => setActive(index)} onClick={() => setActive(index)}
          onKeyDown={event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); setActive(index); } }} />
      </g>)}
      <text x="52" y="199">0</text><text x="346" y="199" textAnchor="middle">{tr('层号', 'Layer ID')}</text><text x="640" y="199" textAnchor="end">{analysis.layer_count - 1}</text>
    </svg>
    <div className="chart-readout">{selected ? <><strong>L{selected.layer}</strong><span>{selected.bpw == null ? tr('此层无该投影', 'No projection in this layer') : `${selected.bpw.toFixed(3)} bpw`}</span>
      <small>{selected.formats.join(' · ')} {selected.parameters > 0 && `· ${parameterLabel(selected.parameters)}`}</small></>
      : <span>{tr('横轴层号，纵轴 bpw；空白表示该层没有此投影。', 'Layer ID on x, bpw on y; blank bars indicate absent projections.')}</span>}</div>
  </section>;
}
