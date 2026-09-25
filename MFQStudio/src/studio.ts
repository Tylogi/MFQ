/** 保留桌面桥接的旧导入路径，具体平台适配集中在共享平台层。 */
export {
  configureStudio,
  isStudio,
  saveStudioCredential,
  selectLocalModelDirectory,
  startLocalStudio,
  studioConfirm,
  studioCredential,
  studioStatus,
} from './shared/platform/studio';
export type { StudioConfig, StudioRuntimeMode, StudioStatus } from './shared/platform/studio';
