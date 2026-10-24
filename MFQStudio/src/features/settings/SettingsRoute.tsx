import { useState } from 'react';
import { ScreenHeader, TMPanel } from '../../app/display';
import { DEFAULT_SETTINGS } from './configuration';
import { SettingsPage } from './SettingsPage';
import { useSettings } from './SettingsProvider';
import { toast } from '../../stores/toastStore';
import { ConcurrencyFields } from '../models/ConcurrencyFields';
import { useConcurrencyPolicy } from '../models/useConcurrencyPolicy';
import { validConcurrency } from '../models/concurrencySettings';

export function SettingsRoute() {
  const { settings, updateSettings, tr } = useSettings();
  const [draft, setDraft] = useState(() => ({ language: settings.language, theme: settings.theme }));
  const [concurrency, setConcurrency] = useState(settings.concurrencyDefaults);
  const [resetVersion, setResetVersion] = useState(0);
  const { policy, error } = useConcurrencyPolicy();
  const valid = policy ? validConcurrency(concurrency, policy)
    : Object.values(concurrency).every(value => value == null);
  function resetSettingsDraft() {
    setDraft({ language: DEFAULT_SETTINGS.language, theme: DEFAULT_SETTINGS.theme });
    setConcurrency(DEFAULT_SETTINGS.concurrencyDefaults);
    setResetVersion(version => version + 1);
  }
  function saveSettings() {
    if (!valid) return;
    updateSettings({ ...draft, concurrencyDefaults: concurrency });
    toast.success(tr('设置已应用', 'Settings applied successfully'));
  }
  return (
    <section className="dashboard-view">
      <ScreenHeader
        title={tr('设置', 'Settings')}
        subtitle={tr('界面语言、主题与新模型的默认并发。', 'Language, theme and concurrency defaults for new models.')}
        trailing={
          <button className="primary" disabled={!valid} onClick={saveSettings} type="button">
            {tr('应用设置', 'Apply settings')}
          </button>
        }
      />
      <TMPanel className="settings-page-panel">
        <h2>{tr('新模型的默认并发', 'Concurrency defaults for new models')}</h2>
        <p>{tr('保存在当前浏览器。新加载的模型沿用；留空时使用当前服务器默认值。',
          'Saved in this browser for new model loads. Empty fields inherit the current server defaults.')}</p>
        <ConcurrencyFields key={resetVersion} value={concurrency} onChange={setConcurrency} policy={policy} />
        {error && <p role="alert">{error}</p>}
      </TMPanel>
      <SettingsPage
        tr={tr}
        canSave={valid}
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
