/** 应用导航与移动侧栏交互，页面路由仍由外壳控制。 */
import { useEffect, type ComponentProps } from 'react';
import { useLocation, useNavigate } from 'react-router';
import { useRuntime } from './RuntimeProvider';
import { useSettings } from '../features/settings/SettingsProvider';
import { useUiStore } from '../stores/uiStore';
import { Icon } from './display';
import { formatNumber } from './formatters';
import { runtimeModelNames } from '../features/runtime/modelSelection';
import { dashboardPath, labPath, resolveStudioLocation, isStudioPath,
  type DashboardPage, type LabPage } from '../navigation';

type NavItem = {
  label: [string, string];
  icon: ComponentProps<typeof Icon>['name'];
  path: string;
  active: boolean;
  current?: boolean;
  count?: number;
};

/** 渲染主导航、运行摘要及移动端侧栏开合控件。 */
export function StudioSidebar() {
  const location = useLocation();
  const navigate = useNavigate();
  const { tr } = useSettings();
  const { runtime, selectedModel: model, models, instances, loading: selectedModelLoading } = useRuntime();
  const sidebarOpen = useUiStore((state) => state.sidebarOpen);
  const closeSidebar = useUiStore((state) => state.closeSidebar);
  const openSidebar = useUiStore((state) => state.openSidebar);
  const currentLocation = resolveStudioLocation(location.pathname);
  const { dashboardPage, labPage } = currentLocation;
  const view = isStudioPath(location.pathname) ? currentLocation.view : 'not-found';
  const availableModelNames = runtimeModelNames(models, instances);
  const selectedModelAvailable = availableModelNames.includes(model);
  const activeRequests = Number(runtime?.active_requests || 0);

  useEffect(() => { closeSidebar(); }, [location.pathname, closeSidebar]);
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key === ',') {
        event.preventDefault();
        navigate('/settings');
      }
      if (event.key === 'Escape') closeSidebar();
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [navigate, closeSidebar]);

  const dashboard = (page: DashboardPage) => dashboardPath(page);
  const lab = (page: LabPage) => labPath(page);
  const groups: { label: [string, string]; items: NavItem[] }[] = [
    { label: ['推理', 'Inference'], items: [
      { label: ['概览', 'Overview'], icon: 'gauge', path: dashboard('overview'),
        active: view === 'dashboard' && dashboardPage === 'overview', current: true,
        count: activeRequests > 0 ? activeRequests : undefined },
      { label: ['模型', 'Models'], icon: 'folder', path: dashboard('models'),
        active: view === 'dashboard' && dashboardPage === 'models', current: true },
      { label: ['服务器', 'Server'], icon: 'server-rack', path: '/runtime',
        active: view === 'dashboard' && dashboardPage === 'connections', current: true },
      { label: ['资源', 'Resources'], icon: 'memory', path: dashboard('cache'),
        active: view === 'dashboard' && dashboardPage === 'cache', current: true },
    ] },
    { label: ['交互', 'Playground'], items: [
      { label: ['对话', 'Chat'], icon: 'chat', path: '/chat', active: view === 'chat', current: true },
    ] },
    { label: ['模型工具', 'Model tools'], items: [
      { label: ['模型仓库', 'Model hub'], icon: 'download', path: lab('models'),
        active: view === 'lab' && labPage === 'models' },
      { label: ['评测与数据集', 'Evaluations'], icon: 'activity', path: lab('evaluations'),
        active: view === 'lab' && labPage === 'evaluations' },
      { label: ['量化工作台', 'Quantization'], icon: 'memory', path: lab('quantization'),
        active: view === 'lab' && labPage === 'quantization' },
    ] },
    { label: ['系统', 'System'], items: [
      { label: ['日志', 'Logs'], icon: 'activity', path: dashboard('logs'),
        active: view === 'dashboard' && dashboardPage === 'logs', current: true },
      { label: ['设置', 'Settings'], icon: 'settings', path: '/settings',
        active: view === 'dashboard' && dashboardPage === 'settings', current: true },
    ] },
  ];

  /** 切换路由后关闭移动侧栏。 */
  function open(path: string) {
    navigate(path);
    closeSidebar();
  }

  return (
    <>
      <aside className={`sidebar ${sidebarOpen ? 'open' : ''}`} id="studio-sidebar">
        <div className="brand">
          <img src="/mfq-mark.svg" alt="" />
          <div><strong>MFQ</strong><span>Studio</span></div>
        </div>
        <div className="sidebar-scroll">
          <nav className="sectioned-nav" aria-label={tr('推理', 'Inference')}>
            {groups.map((group) => (
              <section key={group.label[1]}>
                <div className="sidebar-group-label">{tr(...group.label)}</div>
                {group.items.map((item) => (
                  <button
                    aria-current={item.current && item.active ? 'page' : undefined}
                    className={item.active ? 'active' : ''}
                    key={item.path}
                    onClick={() => open(item.path)}
                    type="button"
                  >
                    <Icon name={item.icon} />
                    {tr(...item.label)}
                    {item.count != null && <span>{formatNumber(item.count)}</span>}
                  </button>
                ))}
              </section>
            ))}
          </nav>
        </div>
        <button className="sidebar-runtime-card" onClick={() => open(dashboard('overview'))} type="button">
          <span className={`runtime-dot ${activeRequests > 0 ? 'busy' : selectedModelAvailable
            ? 'ready' : selectedModelLoading ? 'busy' : 'idle'}`} />
          <span>
            <strong>{model || tr('服务空闲', 'Server idle')}</strong>
            <small>{availableModelNames.length > 1
              ? tr(`${availableModelNames.length} 个模型已加载`,
                `${availableModelNames.length} models loaded`)
              : selectedModelAvailable
                ? `${formatNumber(activeRequests)} ${tr('个活动请求', 'active requests')}`
                : selectedModelLoading
                  ? tr('模型加载中', 'Model loading')
                  : tr('选择模型以开始', 'Choose a model to begin')}</small>
          </span>
          <Icon name="activity" size={14} />
        </button>
      </aside>
      <button aria-controls="studio-sidebar" aria-expanded={sidebarOpen}
        aria-label={tr('打开侧栏', 'Open sidebar')} className="mobile-menu-trigger"
        onClick={openSidebar} type="button">
        <Icon name="menu" size={17} />
      </button>
      <button aria-label={tr('关闭侧栏', 'Close sidebar')}
        className={`mobile-scrim ${sidebarOpen ? 'open' : ''}`}
        onClick={closeSidebar} type="button" />
    </>
  );
}
