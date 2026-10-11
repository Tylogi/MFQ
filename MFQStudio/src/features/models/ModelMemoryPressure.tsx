import { memoryPressureRows, type PressureSystem, type WeightRoles } from './memoryPressure';

export function ModelPressureBars({ weights, roles, system, available, cacheBytes, moe, label, tr }: {
  weights?: number | null;
  roles?: WeightRoles;
  system?: PressureSystem;
  available?: number | null;
  cacheBytes?: number;
  moe?: boolean;
  label: string;
  tr(chinese: string, english: string): string;
}) {
  const rows = memoryPressureRows({ weights, roles, system, available, cacheBytes, moe });
  const bytes = (value?: number | null) => value == null ? '—' : `${(value / 2 ** 30).toFixed(1)} GiB`;
  return <div className="variant-memory-tiers">{rows.map(row => <div key={row.kind}>
    {row.kind !== 'shared' && <small className="variant-memory-caption" title={row.kind === 'ram'
      ? tr('路由专家 + Embedding', 'Routed experts + embedding')
      : rows.length > 1 ? tr('稠密权重 + 已应用 KV', 'Dense weights + applied KV') : tr('整模权重 + 已应用 KV', 'All model weights + applied KV')}>
      {row.kind === 'vram' ? tr('显存', 'VRAM') : 'RAM'} · {bytes(row.required)} / {bytes(row.available)}
    </small>}
    <ModelMemoryPressure required={row.required} available={row.available}
      label={row.kind === 'shared' ? label : row.kind === 'vram' ? tr('显存压力', 'VRAM pressure') : tr('RAM 压力', 'RAM pressure')}
      emptyLabel={tr('无可用内存', 'No available memory')} />
  </div>)}</div>;
}

export function ModelMemoryPressure({ required, available, label, emptyLabel }: {
  required?: number | null;
  available?: number | null;
  label: string;
  emptyLabel: string;
}) {
  const known = required != null && available != null && Number.isFinite(required)
    && Number.isFinite(available) && required >= 0 && available >= 0;
  const exhausted = known && available === 0 && required > 0;
  const percentage = known ? available > 0 ? required / available * 100 : required > 0 ? 100 : 0 : null;
  const percentageLabel = exhausted ? emptyLabel : percentage == null ? '—' : `${percentage.toFixed(1)}%`;
  return <div className="variant-memory-pressure">
    <div aria-label={`${label}: ${percentageLabel}`} aria-valuemax={100} aria-valuemin={0}
      aria-valuenow={percentage == null ? undefined : Math.min(percentage, 100)}
      aria-valuetext={percentageLabel} className="variant-memory-track" role="progressbar">
      <span style={{ width: `${Math.min(percentage ?? 0, 100)}%` }} />
    </div>
    <strong>{percentageLabel}</strong>
  </div>;
}
