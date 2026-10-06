/** Render grouped navigation and the current runtime summary. */
import { useTranslation } from 'react-i18next';
import { useEffect, type ComponentProps } from 'react';
import { useLocation, useNavigate } from 'react-router';
import { useRuntime } from './RuntimeProvider';
import {
  UpdateAvailableBanner,
  useStudioUpdateContext,
} from '../features/settings/UpdateManager';
import { useUiStore } from '../stores/uiStore';
import { Icon } from './display';
import { ModelVendorMark, modelVendor } from './ModelVendorMark';
import { formatNumber } from './formatters';
import { runtimeModelNames } from '../features/runtime/modelSelection';
import { dashboardPath, labPath, resolveStudioLocation, isStudioPath, STUDIO_PATHS,
  type DashboardPage, type LabPage } from '../navigation';

type NavItem = {
  label: string;
  icon: ComponentProps<typeof Icon>['name'];
  path: string;
  active: boolean;
  current?: boolean;
  count?: number;
};

/** Render grouped navigation and the current runtime summary. */
export function StudioSidebar() {
  const location = useLocation();
  const navigate = useNavigate();
  const { t } = useTranslation();
  const studioUpdates = useStudioUpdateContext();
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
  const groups: { label: string; items: NavItem[] }[] = [
    { label: t('app:studioSidebar.inference'), items: [
      { label: t('app:studioSidebar.overview'), icon: 'gauge', path: dashboard('overview'),
        active: view === 'dashboard' && dashboardPage === 'overview', current: true,
        count: activeRequests > 0 ? activeRequests : undefined },
      { label: t('app:studioSidebar.models'), icon: 'folder', path: dashboard('models'),
        active: view === 'dashboard' && dashboardPage === 'models', current: true },
      { label: t('app:studioSidebar.service'), icon: 'server-rack', path: '/runtime',
        active: view === 'dashboard' && dashboardPage === 'connections', current: true },
      { label: t('app:studioSidebar.resources'), icon: 'memory', path: dashboard('cache'),
        active: view === 'dashboard' && dashboardPage === 'cache', current: true },
    ] },
    { label: t('app:studioSidebar.playground'), items: [
      { label: t('app:studioSidebar.chat'), icon: 'chat', path: '/chat', active: view === 'chat', current: true },
    ] },
    { label: t('app:studioSidebar.modelTools'), items: [
      { label: t('app:studioSidebar.modelDownloads'), icon: 'download', path: lab('models'),
        active: view === 'lab' && labPage === 'models' },
      { label: t('app:studioSidebar.evaluations'), icon: 'activity', path: lab('evaluations'),
        active: view === 'lab' && labPage === 'evaluations' },
      { label: t('app:studioSidebar.quantization'), icon: 'memory', path: STUDIO_PATHS.quantizationComingSoon,
        active: view === 'lab' && labPage === 'quantization' },
    ] },
    { label: t('app:studioSidebar.system'), items: [
      { label: t('app:studioSidebar.logs'), icon: 'activity', path: dashboard('logs'),
        active: view === 'dashboard' && dashboardPage === 'logs', current: true },
      { label: t('common:settings'), icon: 'settings', path: '/settings',
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
          <nav className="sectioned-nav" aria-label={t('app:studioSidebar.inference')}>
            {groups.map((group) => (
              <section key={group.items[0].path}>
                <div className="sidebar-group-label">{group.label}</div>
                {group.items.map((item) => (
                  <button
                    aria-current={item.active ? 'page' : undefined}
                    className={item.active ? 'active' : ''}
                    key={item.path}
                    onClick={() => open(item.path)}
                    type="button"
                  >
                    <Icon name={item.icon} />
                    {item.label}
                    {item.count != null && <span>{formatNumber(item.count)}</span>}
                  </button>
                ))}
              </section>
            ))}
          </nav>
        </div>
        <UpdateAvailableBanner
          onOpen={() => open('/settings')}
          status={studioUpdates.status}
          t={t}
        />
        <button className="sidebar-runtime-card" onClick={() => open(dashboard('overview'))} type="button">
          <span className={`runtime-dot ${activeRequests > 0 ? 'busy' : selectedModelAvailable
            ? 'ready' : selectedModelLoading ? 'busy' : 'idle'}`} />
          <span>
            <strong>{model || t('app:studioSidebar.serverIdle')}</strong>
            <small>{availableModelNames.length > 1
              ? t('app:studioSidebar.modelsLoaded', { count: availableModelNames.length })
              : selectedModelAvailable
                ? `${formatNumber(activeRequests)} ${t('app:studioSidebar.activeRequests')}`
                : selectedModelLoading
                  ? t('app:studioSidebar.modelLoading')
                  : t('app:studioSidebar.chooseAModelToBegin')}</small>
          </span>
          {modelVendor(model, model === runtime?.model ? runtime.model_capabilities?.architecture_family || runtime.model_type : undefined)
            ? <ModelVendorMark name={model} size={20}
              architecture={model === runtime?.model ? runtime.model_capabilities?.architecture_family || runtime.model_type : undefined} />
            : <Icon name="activity" size={14} />}
        </button>
      </aside>
      <button aria-controls="studio-sidebar" aria-expanded={sidebarOpen}
        aria-label={t('app:studioSidebar.openSidebar')} className="mobile-menu-trigger"
        onClick={openSidebar} type="button">
        <Icon name="menu" size={17} />
      </button>
      <button aria-label={t('app:studioSidebar.closeSidebar')}
        className={`mobile-scrim ${sidebarOpen ? 'open' : ''}`}
        onClick={closeSidebar} type="button" />
    </>
  );
}
