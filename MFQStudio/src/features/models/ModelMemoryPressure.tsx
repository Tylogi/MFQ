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
