/** 保留桌面桥接的旧导入路径，具体平台适配集中在共享平台层。 */
export {
  configureStudio,
  deleteStudioVersion,
  downloadStudioVersion,
  installStudioVersion,
  isStudio,
  openStudioExternal,
  saveStudioCredential,
  selectLocalModelDirectory,
  setAutomaticStudioUpdates,
  startLocalStudio,
  studioConfirm,
  studioCredential,
  studioUpdateProgress,
  studioUpdateStatus,
  studioStatus,
} from './shared/platform/studio';
export type {
  InstalledStudioVersion,
  StudioConfig,
  StudioRelease,
  StudioReleaseAsset,
  StudioRuntimeMode,
  StudioStatus,
  StudioUpdateProgress,
  StudioUpdateStatus,
} from './shared/platform/studio';
