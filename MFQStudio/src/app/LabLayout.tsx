/** 模型工具复用页面标题，导航由侧栏统一提供。 */
import { Outlet, useLocation } from 'react-router';
import { useSettings } from '../features/settings/SettingsProvider';
import { ScreenHeader } from './display';

const tools = [
  { path: '/model-hub', zh: '模型下载', en: 'Model downloads',
    detail: ['浏览模型，选择适合设备的精度版本。', 'Find models and precision tiers for your device.'] },
  { path: '/evaluations', zh: '评测与数据集', en: 'Evaluations',
    detail: ['管理评测任务与校准数据集。', 'Manage evaluations and calibration datasets.'] },
  { path: '/quantization', zh: '量化工作台', en: 'Quantization',
    detail: ['从校准到导出，管理模型量化任务。', 'Manage model quantization, from calibration to export.'] },
];

/** 为模型来源、评测和量化页面提供一致的可导航容器。 */
export function LabLayout() {
  const { tr } = useSettings();
  const { pathname } = useLocation();
  const current = tools.find((item) => item.path === pathname) ?? tools[0];
  return (
    <section className="dashboard-view lab-view">
      <ScreenHeader title={tr(current.zh, current.en)}
        subtitle={tr(current.detail[0], current.detail[1])} />
      <Outlet />
    </section>
  );
}
