import { Fragment, useState } from 'react';
import type { HubModelVariant, ModelCacheProfile, ModelConfigurationStatus } from '../../shared/api/types';
import { CaretDownIcon } from '@phosphor-icons/react';
import { cacheBreakdown, cacheComponentLabel, cacheUsageBytes, estimateCacheBytes, kvOnlyProfile } from './cacheData';
import type { CacheBreakdown } from './cacheData';

export { cacheUsageBytes, estimateCacheBytes } from './cacheData';

type Translate = (chinese: string, english: string) => string;

function memoryLabel(bytes: number): string {
  return bytes < 2 ** 30 ? `${(bytes / 2 ** 20).toFixed(1)} MiB` : `${(bytes / 2 ** 30).toFixed(2)} GiB`;
}

export function withCacheMemory(variants: HubModelVariant[], bytes: number): HubModelVariant[] {
  if (bytes <= 0) return variants;
  return variants.map((variant) => {
    const original = variant.configuration;
    const required = original.required_memory_bytes;
    if (required == null) return variant;
    const configuration = { ...original, required_memory_bytes: required + bytes };
    return { ...variant, configuration };
  });
}

export function plannedConfiguration(variants: HubModelVariant[], fallback: ModelConfigurationStatus, tr: Translate): ModelConfigurationStatus {
  const budget = fallback.available_memory_bytes;
  const candidates = variants.filter((variant) => (variant.format === 'mfq' || variant.format === 'hf')
    && variant.configuration.status !== 'unknown' && variant.configuration.required_memory_bytes != null);
  if (!candidates.length || budget == null || budget <= 0) return fallback;
  const requirements = candidates.map((variant) => variant.configuration.required_memory_bytes!);
  const minimum = Math.min(...requirements);
  const fits = requirements.filter((bytes) => bytes <= budget).length;
  const recommendation = fits === requirements.length ? 'three_stars' : fits * 2 > requirements.length ? 'two_stars'
    : fits > 0 ? 'one_star' : budget >= minimum * 0.7 ? 'caution' : 'not_recommended';
  return {
    status: fits > 0 ? 'recommended' : 'warning', recommendation,
    required_memory_bytes: minimum, available_memory_bytes: budget,
    reasons: [tr(`预计权重常驻 + 规划 KV/递推状态：${fits}/${requirements.length} 个档位在预算以内。`, `Resident weights + planned KV/recurrent state: ${fits}/${requirements.length} tiers fit the budget.`)],
  };
}

function CacheRows({ items, depth = 0, tr }: { items: CacheBreakdown[]; depth?: number; tr: Translate }) {
  return items.map(item => <Fragment key={item.id}><tr className={depth === 0 ? 'kv-cache-group-row' : undefined}>
    <th scope="row" style={{ paddingLeft: `${12 + depth * 16}px` }}>{depth === 0 ? `${item.name} ${tr('总缓存', 'total cache')}` : cacheComponentLabel(item, tr)}
      {depth === 0 && item.layers != null && <small>{item.layers} {tr('层', 'layers')}</small>}</th>
    <td>{memoryLabel(item.usage)}</td><td>{memoryLabel(item.budget)}</td>
  </tr>{item.children.length > 0 && <CacheRows items={item.children} depth={depth + 1} tr={tr} />}</Fragment>);
}

export function KvCachePlanner({ profile: sourceProfile, appliedContext, onApply, tr }: {
  profile?: ModelCacheProfile | null;
  appliedContext?: number;
  onApply?(context: number | undefined): void;
  tr: Translate;
}) {
  const profile = sourceProfile ? kvOnlyProfile(sourceProfile) : null;
  const [draft, setDraft] = useState(String(appliedContext ?? Math.min(4096, profile?.max_context ?? 4096)));
  const [hoverContext, setHoverContext] = useState<number | null>(null);
  const [selectedGroup, setSelectedGroup] = useState('all');
  const context = Number(draft);
  const valid = !!profile && Number.isSafeInteger(context) && context >= 1 && context <= profile.max_context;
  const preview = profile && valid ? estimateCacheBytes(profile, context) : null;
  const end = profile?.max_context ?? 32768;
  const start = Math.min(1024, 2 ** Math.floor(Math.log2(valid ? context : end)), end);
  const logStart = Math.log2(start);
  const logSpan = Math.max(1, Math.log2(end) - logStart);
  const ticks = [];
  for (let tokens = start; tokens <= end; tokens *= 2) ticks.push(tokens);
  if (ticks[ticks.length - 1] !== end) ticks.push(end);
  const groups = profile && valid ? cacheBreakdown(profile, context) : [];
  const selected = groups.find(item => item.id === selectedGroup) ?? (groups.length === 1 ? groups[0] : null);
  const children = selected ? selected.children.length > 1 ? selected.children : [] : groups;
  const colors = ['var(--accent)', '#b56b79', '#498875', '#a58745', '#8272ad'];
  const curves = [{ id: 'total', label: tr('总缓存', 'Total cache'), color: colors[0] }, ...children.map((item, index) => ({ id: item.id,
    label: selected ? cacheComponentLabel(item, tr) : item.name, color: colors[(index % (colors.length - 1)) + 1] }))];
  function curveBytes(tokens: number) {
    if (!profile) return [];
    const values = cacheBreakdown(profile, tokens);
    const group = selected ? values.find(item => item.id === selected.id) : null;
    const items = group ? group.children.length > 1 ? group.children : [] : values;
    return [group?.usage ?? cacheUsageBytes(profile, tokens), ...items.map(item => item.usage)];
  }
  const samples = profile ? Array.from({ length: 129 }, (_, index) => {
    const tokens = start * (end / start) ** (index / 128);
    return { tokens, values: curveBytes(tokens) };
  }) : [];
  const highest = Math.max(1, ...samples.flatMap(sample => sample.values));
  const x = (tokens: number) => 54 + (Math.log2(tokens) - logStart) / logSpan * 390;
  const y = (bytes: number) => 148 - bytes / highest * 116;
  const paths = curves.map((_, curve) => samples.map((sample, index) => `${index ? 'L' : 'M'} ${x(sample.tokens)} ${y(sample.values[curve] ?? 0)}`).join(' '));
  const pointContext = hoverContext ?? (valid ? context : null);
  const pointValues = pointContext != null ? curveBytes(pointContext) : [];
  const pointBytes = pointValues[0] ?? null;
  function chartContext(element: SVGSVGElement, clientX: number, clientY: number): number | null {
    const rect = element.getBoundingClientRect();
    if (!rect.width || !rect.height) return null;
    const px = (clientX - rect.left) / rect.width * 460;
    const py = (clientY - rect.top) / rect.height * 174;
    if (px < 54 || px > 444 || py < 24 || py > 148) return null;
    return Math.max(start, Math.min(end, Math.round(2 ** (logStart + (px - 54) / 390 * logSpan))));
  }
  function selectContext(tokens: number) {
    setDraft(String(tokens));
    setHoverContext(tokens);
  }
  return (
    <details className="kv-cache-planner">
      <summary><span>{tr('KV Cache 曲线与计算器', 'KV Cache curve & calculator')}</span><span className="kv-cache-summary-end">{profile && appliedContext != null && <small>{appliedContext.toLocaleString()} ctx · {memoryLabel(estimateCacheBytes(profile, appliedContext))}</small>}<CaretDownIcon size={14} aria-hidden="true" /></span></summary>
      {profile ? <div className="kv-cache-planner-body">
        <form className="kv-cache-controls" onSubmit={(event) => { event.preventDefault(); if (valid) onApply?.(context); }}>
          <label>{tr('预期 ctx', 'Expected ctx')}<input type="number" min={1} max={profile.max_context} step={1} value={draft} onChange={(event) => setDraft(event.target.value)} /></label>
          {groups.length > 1 && <label>{tr('缓存类型', 'Cache type')}<select value={selectedGroup} onChange={event => setSelectedGroup(event.target.value)}>
            <option value="all">{tr('全部缓存', 'All caches')}</option>{groups.map(item => <option key={item.id} value={item.id}>{item.name}</option>)}</select></label>}
          {onApply && <><button disabled={!valid} type="submit">{tr('应用', 'Apply')}</button>
          <button disabled={appliedContext == null} onClick={() => onApply(undefined)} type="button">{tr('清除', 'Clear')}</button></>}
        </form>
        {!valid && <p role="alert">{tr('ctx 需为整数，范围', 'ctx must be an integer in')} 1–{profile.max_context.toLocaleString()}。</p>}
        <div className="kv-cache-estimate"><span>{tr('预算估计（含预分配）', 'Budget estimate (including preallocation)')}</span><strong>{preview == null ? '—' : memoryLabel(preview)}</strong></div>
        <div className="kv-cache-chart-shell">
        <svg className="kv-cache-chart" viewBox="0 0 460 174" role="img" aria-label={tr('KV Cache 占用随上下文长度变化', 'KV Cache memory by context length')}
          onMouseMove={(event) => setHoverContext(chartContext(event.currentTarget, event.clientX, event.clientY))}
          onMouseLeave={() => setHoverContext(null)}
          onClick={(event) => { const tokens = chartContext(event.currentTarget, event.clientX, event.clientY); if (tokens != null) selectContext(tokens); }}>
          {[0, 0.5, 1].map((fraction) => <g key={fraction}><line x1={54} x2={444} y1={y(highest * fraction)} y2={y(highest * fraction)} /><text x={48} y={y(highest * fraction) + 3} textAnchor="end">{(highest * fraction / 2 ** 30).toFixed(2)}</text></g>)}
          <text x={54} y={18}>GiB</text>
          <path d={`${paths[0]} L ${x(end)} 148 L 54 148 Z`} className="kv-cache-chart-fill" />
          {curves.map((curve, index) => <path key={curve.id} d={paths[index]} className="kv-cache-chart-line" style={{ stroke: curve.color }} data-cache-curve={curve.id} />)}
          {ticks.map((tokens) => <line key={tokens} x1={x(tokens)} x2={x(tokens)} y1={148} y2={154} />)}
          <text x={444} y={170} textAnchor="end">{tr('ctx（每格 ×2）', 'ctx (×2 steps)')}</text>
          {pointContext != null && pointBytes != null && <g>
            <line className="kv-cache-crosshair" x1={x(pointContext)} x2={x(pointContext)} y1={28} y2={148} />
            <line className="kv-cache-crosshair" x1={54} x2={x(pointContext)} y1={y(pointBytes)} y2={y(pointBytes)} />
            <circle cx={x(pointContext)} cy={y(pointBytes)} r={4} />
          </g>}
        </svg>
        {hoverContext != null && pointBytes != null && <div className="kv-cache-tooltip" role="tooltip" style={{ left: `${Math.max(22, Math.min(78, x(hoverContext) / 460 * 100))}%`, top: `${Math.max(62, y(pointBytes) - 8) / 174 * 100}%` }}>
          <span>{hoverContext.toLocaleString()} ctx</span><strong>{(pointBytes / 2 ** 30).toFixed(3)} GiB</strong>
          {curves.slice(1).map((curve, index) => <span key={curve.id}>{curve.label} · {((pointValues[index + 1] ?? 0) / 2 ** 30).toFixed(3)} GiB</span>)}
        </div>}
        </div>
        <div className="kv-cache-ruler" aria-label={tr('ctx 刻度', 'ctx ticks')}>
          {ticks.map((tokens) => <button key={tokens} type="button" aria-label={`ctx ${tokens}`} aria-pressed={valid && context === tokens}
            style={{ left: `${x(tokens) / 460 * 100}%` }}
            onMouseEnter={() => setHoverContext(tokens)} onMouseLeave={() => setHoverContext(null)}
            onFocus={() => setHoverContext(tokens)} onBlur={() => setHoverContext(null)} onClick={() => selectContext(tokens)}>
            <span className="kv-cache-tick-full">{tokens}</span><span className="kv-cache-tick-short">{tokens < 16384 ? tokens : `${tokens / 1024}k`}</span>
          </button>)}
        </div>
        <div className="kv-cache-legend">{curves.map(curve => <span key={curve.id}><i style={{ background: curve.color }} />{curve.label}</span>)}</div>
        {groups.length > 0 && <div className="kv-cache-breakdown"><div className="kv-cache-breakdown-heading"><strong>{tr('缓存明细', 'Cache breakdown')}</strong><span>{context.toLocaleString()} ctx</span></div>
          <div className="kv-cache-table-scroll"><table><thead><tr><th>{tr('组件', 'Component')}</th><th>{tr('理论占用', 'Theoretical usage')}</th><th>{tr('含预分配', 'Including preallocation')}</th></tr></thead>
            <tbody><CacheRows items={selected ? [selected] : groups} tr={tr} /></tbody></table></div></div>}
        <p>{tr('单个上下文的 KV / Indexer 缓存估算，预算含预分配。点击曲线或刻度选择 ctx。', 'KV / indexer cache estimates for one context; the budget includes preallocation. Click the curve or a tick to select ctx.')}
          {onApply && ` ${tr('应用仅更新下载规划。', 'Apply changes download planning only.')}`}</p>
      </div> : <p className="kv-cache-unavailable">{tr('模型缓存结构信息不足，暂无法可靠估算。', 'Insufficient cache structure metadata for a reliable estimate.')}</p>}
    </details>
  );
}
