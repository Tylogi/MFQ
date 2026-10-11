import { useEffect, type ComponentProps } from 'react';
import { useLocation, useNavigate } from 'react-router';
import { useRuntime } from './RuntimeProvider';
import { useSettings } from '../features/settings/SettingsProvider';
import {
  useStudioUpdateContext,
} from '../features/settings/UpdateManager';
import { useUiStore } from '../stores/uiStore';
import { Icon } from './display';
import { formatNumber } from './formatters';
import { studioBuild } from '../features/settings/releases';
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

export function StudioSidebar() {
  const location = useLocation();
  const navigate = useNavigate();
  const { tr } = useSettings();
  const studioUpdates = useStudioUpdateContext();
  const version = studioUpdates.status?.current_version || studioBuild.version;
  const releaseBuild = studioUpdates.status?.current_release ?? studioBuild.release;
  const revision = version.split('+dev.')[1]?.split('.')[0];
  const { runtime } = useRuntime();
  const sidebarOpen = useUiStore((state) => state.sidebarOpen);
  const closeSidebar = useUiStore((state) => state.closeSidebar);
  const openSidebar = useUiStore((state) => state.openSidebar);
  const currentLocation = resolveStudioLocation(location.pathname);
  const { dashboardPage, labPage } = currentLocation;
  const view = isStudioPath(location.pathname) ? currentLocation.view : 'not-found';
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
      { label: ['服务', 'Service'], icon: 'server-rack', path: '/runtime',
        active: view === 'dashboard' && dashboardPage === 'connections', current: true },
      { label: ['资源', 'Resources'], icon: 'memory', path: dashboard('cache'),
        active: view === 'dashboard' && dashboardPage === 'cache', current: true },
    ] },
    { label: ['交互', 'Playground'], items: [
      { label: ['对话', 'Chat'], icon: 'chat', path: '/chat', active: view === 'chat', current: true },
      { label: ['应用', 'Applications'], icon: 'link', path: '/applications', active: view === 'applications' },
    ] },
    { label: ['工具', 'Tools'], items: [
      { label: ['分析', 'Analysis'], icon: 'search', path: lab('analysis'),
        active: view === 'lab' && labPage === 'analysis' },
      { label: ['下载', 'Downloads'], icon: 'download', path: lab('models'),
        active: view === 'lab' && labPage === 'models' },
      { label: ['测评', 'Evaluations'], icon: 'activity', path: lab('evaluations'),
        active: view === 'lab' && labPage === 'evaluations' },
      { label: ['量化', 'Quantization'], icon: 'tools', path: lab('quantization'),
        active: view === 'lab' && labPage === 'quantization' },
    ] },
    { label: ['系统', 'System'], items: [
      { label: ['日志', 'Logs'], icon: 'scroll', path: dashboard('logs'),
        active: view === 'dashboard' && dashboardPage === 'logs', current: true },
      { label: ['设置', 'Settings'], icon: 'settings', path: '/settings',
        active: view === 'dashboard' && dashboardPage === 'settings', current: true },
    ] },
  ];

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
                    aria-current={item.active ? 'page' : undefined}
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
        <button className={`sidebar-version-card ${location.pathname === '/versions' ? 'active' : ''}`}
          aria-current={location.pathname === '/versions' ? 'page' : undefined}
          aria-label={tr('打开版本管理', 'Open version manager')}
          onClick={() => open('/versions')} type="button">
          <span>
            <strong title={version}>v{version.split('+')[0]}</strong>
            <small>{studioUpdates.busy?.startsWith('download:') ? tr('正在下载更新', 'Downloading update')
              : studioUpdates.status?.update_available ? tr('有新 Release', 'New Release available')
              : releaseBuild ? tr('Release · 版本管理', 'Release · Versions')
                : `${tr('开发版', 'Dev')} · ${revision || tr('版本管理', 'Versions')}`}</small>
          </span>
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
