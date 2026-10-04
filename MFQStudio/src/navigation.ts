/**
 * Define MFQ Studio navigation and maintain shared route paths for desktop and browser clients.
 */

export type ViewName = 'chat' | 'dashboard' | 'lab';
export type DashboardPage = 'overview' | 'models' | 'connections' | 'cache' | 'logs' | 'settings';
export type LabPage = 'models' | 'evaluations' | 'quantization';

export interface StudioLocation {
  view: ViewName;
  dashboardPage: DashboardPage;
  labPage: LabPage;
}

export const STUDIO_PATHS = {
  overview: '/',
  chat: '/chat',
  models: '/models',
  server: '/runtime',
  resources: '/resources',
  modelHub: '/model-hub',
  evaluations: '/evaluations',
  quantization: '/quantization',
  logs: '/logs',
  settings: '/settings',
} as const;

const DASHBOARD_PATHS: Record<DashboardPage, string> = {
  overview: STUDIO_PATHS.overview,
  models: STUDIO_PATHS.models,
  connections: STUDIO_PATHS.server,
  cache: STUDIO_PATHS.resources,
  logs: STUDIO_PATHS.logs,
  settings: STUDIO_PATHS.settings,
};

const LAB_PATHS: Record<LabPage, string> = {
  models: STUDIO_PATHS.modelHub,
  evaluations: STUDIO_PATHS.evaluations,
  quantization: STUDIO_PATHS.quantization,
};

/** Normalize Studio paths, handling trailing slashes supplied by browsers or Tauri. */
export function normalizeStudioPath(pathname: string): string {
  return pathname.length > 1 ? pathname.replace(/\/+$/, '') : pathname;
}

/** Determine whether a path is a registered product page; unknown paths are handled by the 404 route. */
export function isStudioPath(pathname: string): boolean {
  return Object.values(STUDIO_PATHS).some((path) => path === normalizeStudioPath(pathname));
}

/** Resolve which Studio product page to display from the current URL. */
export function resolveStudioLocation(pathname: string): StudioLocation {
  const normalizedPath = normalizeStudioPath(pathname);
  if (normalizedPath === STUDIO_PATHS.chat) {
    return { view: 'chat', dashboardPage: 'overview', labPage: 'models' };
  }

  const dashboardPage = (Object.entries(DASHBOARD_PATHS).find(([, path]) => path === normalizedPath)?.[0]
    ?? 'overview') as DashboardPage;
  const labPage = Object.entries(LAB_PATHS).find(([, path]) => path === normalizedPath)?.[0] as LabPage | undefined;

  return labPage
    ? { view: 'lab', dashboardPage: 'overview', labPage }
    : { view: 'dashboard', dashboardPage, labPage: 'models' };
}

/** Return the stable route for the specified Dashboard page. */
export function dashboardPath(page: DashboardPage): string {
  return DASHBOARD_PATHS[page];
}

/** Return the stable route for the specified model-tools page. */
export function labPath(page: LabPage): string {
  return LAB_PATHS[page];
}
