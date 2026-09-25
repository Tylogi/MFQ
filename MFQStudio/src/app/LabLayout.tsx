/** 模型工具共享标题和二级导航，仅承载布局，不管理任务或评测状态。 */
import { Outlet, useLocation, useNavigate } from 'react-router';
import { useSettings } from '../features/settings/SettingsProvider';
import { Icon } from './display';

const tools = [
  { path: '/model-hub', zh: '模型仓库', en: 'Model hub', icon: 'download' as const },
  { path: '/evaluations', zh: '评测与数据集', en: 'Evaluations', icon: 'activity' as const },
  { path: '/quantization', zh: '量化工作台', en: 'Quantization', icon: 'memory' as const },
];

/** 为模型来源、评测和量化页面提供一致的可导航容器。 */
export function LabLayout() {
  const { tr } = useSettings();
  const { pathname } = useLocation();
  const navigate = useNavigate();
  const current = tools.find((item) => item.path === pathname) ?? tools[0];
  return (
    <section className="dashboard-view lab-view">
      <div className="page-heading">
        <div>
          <h1>{tr(current.zh, current.en)}</h1>
        </div>
      </div>
      <nav className="workspace-tabs" aria-label={tr('模型工具', 'Model tools')}>
        {tools.map((item) => (
          <button
            key={item.path}
            aria-current={item.path === pathname ? 'page' : undefined}
            className={item.path === pathname ? 'active' : ''}
            onClick={() => navigate(item.path)}
            type="button"
          >
            <Icon name={item.icon} size={14} />
            {tr(item.zh, item.en)}
          </button>
        ))}
      </nav>
      <Outlet />
    </section>
  );
}
