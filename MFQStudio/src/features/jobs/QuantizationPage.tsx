/** Render the quantization workspace panels. */
import { useTranslation } from 'react-i18next';

/** Render the quantization workspace panels. */
export function QuantizationPage() {
  const { t } = useTranslation();
  return (
    <div className="quantization-empty-grid" aria-label={t('jobs:quantizationPage.quantizationWorkspace')}>
      {['imatrix', 'lineage', 'builder', 'history'].map((id) => (
        <section className="dashboard-panel quantization-empty-panel" key={id} aria-hidden="true" />
      ))}
    </div>
  );
}
