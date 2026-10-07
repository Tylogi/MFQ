import { useId, useMemo, useRef, useState } from 'react';
import type { KeyboardEvent } from 'react';
import type { CheckpointAnalysis } from '../../shared/api/types';
import { Icon } from '../../app/display';
import { Dialog } from '../../shared/ui/Dialog';
import type { Translate } from './analysisData';
import { architectureReference, architectureSchedule, architectureViews, type ArchitectureView } from './architectureData';

function ArchitectureFlow({ view, tr, onSelect }: { view: ArchitectureView; tr: Translate; onSelect: (id: string) => void }) {
  const marker = useId().replaceAll(':', '');
  const nodes = new Map(view.nodes.map(node => [node.id, node]));
  return <svg className="architecture-flow" viewBox={`0 0 360 ${view.height}`} role="img" aria-label={`${view.title} · ${tr('架构细节', 'architecture detail')}`}>
    <defs><marker id={marker} markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0L7 3.5L0 7" className="architecture-arrowhead" /></marker></defs>
    {view.frames?.map((frame, index) => <g key={index}><rect x={frame.x} y={frame.y} width={frame.width} height={frame.height} rx="12" className="architecture-repeat" />
      <text x={frame.x + 13} y={frame.y + 18} className="architecture-repeat-label">{frame.title}</text></g>)}
    {view.edges.map((edge, index) => {
      const from = nodes.get(edge.from)!, to = nodes.get(edge.to)!;
      const start = from.y + from.height, middle = (start + to.y) / 2;
      return <path key={index} d={edge.path ?? `M${from.x + from.width / 2} ${start}V${middle}H${to.x + to.width / 2}V${to.y - 3}`}
        className={edge.residual ? 'architecture-residual' : undefined} markerEnd={`url(#${marker})`} />;
    })}
    {view.nodes.map(node => <g key={node.id} className={node.action ? 'architecture-action' : undefined}
      {...(node.action ? { role: 'button', tabIndex: 0, 'aria-label': `${tr('查看', 'Inspect')} ${node.title}`,
        onClick: () => onSelect(node.action!), onKeyDown: (event: KeyboardEvent<SVGGElement>) => {
          if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); onSelect(node.action!); }
        } } : {})}>
      <rect x={node.x} y={node.y} width={node.width} height={node.height} rx="8" className={node.accent ? 'architecture-accent' : undefined} />
      <text x={node.x + node.width / 2} y={node.y + (node.detail ? 21 : 23)} textAnchor="middle" className="architecture-node-title">{node.title}</text>
      {node.detail && <text x={node.x + node.width / 2} y={node.y + 39} textAnchor="middle" className="architecture-node-detail">{node.detail}</text>}
    </g>)}
  </svg>;
}

export function ArchitectureDiagram({ analysis, tr }: { analysis: CheckpointAnalysis; tr: Translate }) {
  const views = useMemo(() => architectureViews(analysis, tr), [analysis, tr]);
  const [choice, setChoice] = useState('overview');
  const [expanded, setExpanded] = useState(false);
  const trigger = useRef<HTMLButtonElement>(null);
  const view = views.find(item => item.id === choice) ?? views[0];
  const reference = architectureReference(analysis);
  const schedule = architectureSchedule(analysis);
  const prefix = useId();
  function navigation(id: string) {
    return <div className="architecture-tabs" role="tablist" aria-label={tr('架构组件', 'Architecture components')}>
      {views.map(item => <button key={item.id} type="button" role="tab" id={`${id}-${item.id}`} aria-selected={view.id === item.id}
        aria-controls={`${id}-diagram`} onClick={() => setChoice(item.id)}>{item.title}</button>)}</div>;
  }
  function content(id: string) {
    return <div role="tabpanel" id={`${id}-diagram`} aria-labelledby={`${id}-${view.id}`}>
      <div className="architecture-canvas"><ArchitectureFlow view={view} tr={tr} onSelect={setChoice} /></div>
      <p className="architecture-description">{view.description}</p>
    </div>;
  }
  return <section className="analysis-panel analysis-architecture">
    <div className="analysis-panel-heading"><h2>{tr('架构图', 'Architecture')}</h2><button ref={trigger} type="button" className="architecture-expand"
      aria-label={tr('放大架构图', 'Expand architecture')} onClick={() => setExpanded(true)}><Icon name="search" size={14} />{tr('放大', 'Expand')}</button></div>
    {schedule && <div className="architecture-schedule">{schedule}</div>}
    {navigation(prefix)}
    {content(prefix)}
    <div className="architecture-source"><span>{tr('按当前检查点绘制', 'Drawn for this checkpoint')}</span>
      {reference && <a href={reference.href} target="_blank" rel="noreferrer">{tr('官方模型卡', 'Official model card')} ↗</a>}</div>
    <Dialog open={expanded} onOpenChange={setExpanded} title={tr('架构细节', 'Architecture details')} description={analysis.name}
      closeLabel={tr('关闭架构图', 'Close architecture')} className="architecture-dialog" returnFocusRef={trigger}>
      <div className="architecture-dialog-body">{navigation(`${prefix}-expanded`)}{content(`${prefix}-expanded`)}</div>
    </Dialog>
  </section>;
}
