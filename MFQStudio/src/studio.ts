/** Preserve the legacy desktop-bridge import path; platform-specific adapters live in the shared platform layer. */
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
