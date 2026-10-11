import { ScreenHeader } from '../../app/display';
import { UpdateManager, useStudioUpdateContext } from './UpdateManager';
import { useSettings } from './SettingsProvider';

export function VersionsPage() {
  const { tr } = useSettings();
  const updates = useStudioUpdateContext();
  return <section className="dashboard-view versions-page">
    <ScreenHeader title={tr('版本管理', 'Version manager')} subtitle={tr('浏览正式版、更新说明和本地版本。', 'Browse Releases, version notes, and local versions.')} />
    <UpdateManager {...updates} tr={tr} />
  </section>;
}
