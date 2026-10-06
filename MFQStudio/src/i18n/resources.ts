/** Bundle all locale catalogs so Web and desktop interfaces also translate offline. */
import enApp from './locales/en/app.json';
import enChat from './locales/en/chat.json';
import enCommon from './locales/en/common.json';
import enConnections from './locales/en/connections.json';
import enEvaluations from './locales/en/evaluations.json';
import enJobs from './locales/en/jobs.json';
import enModels from './locales/en/models.json';
import enRuntime from './locales/en/runtime.json';
import enSettings from './locales/en/settings.json';
import zhApp from './locales/zh-CN/app.json';
import zhChat from './locales/zh-CN/chat.json';
import zhCommon from './locales/zh-CN/common.json';
import zhConnections from './locales/zh-CN/connections.json';
import zhEvaluations from './locales/zh-CN/evaluations.json';
import zhJobs from './locales/zh-CN/jobs.json';
import zhModels from './locales/zh-CN/models.json';
import zhRuntime from './locales/zh-CN/runtime.json';
import zhSettings from './locales/zh-CN/settings.json';

export const namespaces = [
  'common',
  'app',
  'chat',
  'connections',
  'evaluations',
  'jobs',
  'models',
  'runtime',
  'settings',
] as const;

export const resources = {
  en: {
    common: enCommon,
    app: enApp,
    chat: enChat,
    connections: enConnections,
    evaluations: enEvaluations,
    jobs: enJobs,
    models: enModels,
    runtime: enRuntime,
    settings: enSettings,
  },
  'zh-CN': {
    common: zhCommon,
    app: zhApp,
    chat: zhChat,
    connections: zhConnections,
    evaluations: zhEvaluations,
    jobs: zhJobs,
    models: zhModels,
    runtime: zhRuntime,
    settings: zhSettings,
  },
};
