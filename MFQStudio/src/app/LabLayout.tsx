/** Share page headers across model tools while centralizing navigation in the sidebar. */
import { useTranslation } from 'react-i18next';
import { Outlet, useLocation } from 'react-router';
import { ScreenHeader } from './display';



/** Provide a consistent navigable container for model sources, evaluations, and quantization pages. */
export function LabLayout() {
  const { t } = useTranslation();
const tools = [
  { path: '/model-hub', title: t('app:labLayout.modelDownloads'),
    detail: t('app:labLayout.findModelsAndPrecisionTiersForYourDevice') },
  { path: '/evaluations', title: t('app:labLayout.evaluations'),
    detail: t('app:labLayout.manageEvaluationsAndCalibrationDatasets') },
  { path: '/quantization', title: t('app:labLayout.quantization'),
    detail: t('app:labLayout.manageModelQuantizationFromCalibrationToExport') },
];
  const { pathname } = useLocation();
  const current = tools.find((item) => item.path === pathname) ?? tools[0];
  return (
    <section className="dashboard-view lab-view">
      <ScreenHeader title={current.title}
        subtitle={current.detail} />
      <Outlet />
    </section>
  );
}
