import { useSettings } from '../settings/SettingsProvider';

export function QuantizationPage() {
  const { tr } = useSettings();
  return (
    <div className="quantization-empty-grid" aria-label={tr('量化工作台', 'Quantization workspace')}>
      {['imatrix', 'lineage', 'builder', 'history'].map((id) => (
        <section className="dashboard-panel quantization-empty-panel" key={id} aria-hidden="true" />
      ))}
    </div>
  );
}
