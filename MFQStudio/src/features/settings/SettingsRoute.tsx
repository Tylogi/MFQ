import { useState } from 'react';
import { ScreenHeader } from '../../app/display';
import { DEFAULT_SETTINGS } from './configuration';
import { SettingsPage } from './SettingsPage';
import { useSettings } from './SettingsProvider';
import { toast } from '../../stores/toastStore';

export function SettingsRoute() {
  const { settings, updateSettings, tr } = useSettings();
  const [draft, setDraft] = useState(() => ({ language: settings.language, theme: settings.theme }));
  function resetSettingsDraft() {
    setDraft({ language: DEFAULT_SETTINGS.language, theme: DEFAULT_SETTINGS.theme });
  }
  function saveSettings() {
    updateSettings(draft);
    toast.success(tr('设置已应用', 'Settings applied successfully'));
  }
  return (
    <section className="dashboard-view">
      <ScreenHeader
        title={tr('设置', 'Settings')}
        subtitle={tr('界面语言与主题。', 'Interface language and theme.')}
        trailing={
          <button className="primary" onClick={saveSettings} type="button">
            {tr('应用设置', 'Apply settings')}
          </button>
        }
      />
      <SettingsPage
        tr={tr}
        settingsDraft={draft}
        setSettingsDraft={setDraft}
        actions={{
          resetSettingsDraft,
          saveSettings,
        }}
      />
    </section>
  );
}
