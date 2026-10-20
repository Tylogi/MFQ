import type { Dispatch, SetStateAction } from 'react';
import { SectionLabel, SettingRow, TMPanel } from '../../app/display';
import type { GenerationSettings, UiLanguage, UiTheme } from './configuration';

export type AppearanceSettings = Pick<GenerationSettings, 'language' | 'theme'>;

export interface SettingsActions {
  resetSettingsDraft: () => void;
  saveSettings: () => void;
}
interface SettingsPageProps {
  tr: (zh: string, en: string) => string;
  settingsDraft: AppearanceSettings;
  setSettingsDraft: Dispatch<SetStateAction<AppearanceSettings>>;
  actions: SettingsActions;
}

export function SettingsPage({ tr, settingsDraft, setSettingsDraft, actions }: SettingsPageProps) {
  const { resetSettingsDraft, saveSettings } = actions;
  return (
    <div className="settings-page">
      <div className="settings-page-section">
        <SectionLabel title={tr("外观", "Appearance")} />
        <TMPanel className="settings-page-panel">
          <SettingRow
            title={tr("界面语言", "Interface language")}
            detail={tr("选择 MFQ Studio 的显示语言。", "Choose the display language for MFQ Studio.")}
            trailing={<select aria-label={tr("界面语言", "Interface language")} onChange={(event) => setSettingsDraft((current) => ({ ...current, language: event.target.value as UiLanguage }))} value={settingsDraft.language}><option value="system">{tr("跟随系统", "System")}</option><option value="zh-CN">简体中文</option><option value="en">English</option></select>}
          />
          <SettingRow
            title={tr("主题", "Theme")}
            detail={tr("跟随系统，或固定使用浅色或深色外观。", "Follow the system or use a fixed light or dark appearance.")}
            trailing={<select aria-label={tr("主题", "Theme")} onChange={(event) => setSettingsDraft((current) => ({ ...current, theme: event.target.value as UiTheme }))} value={settingsDraft.theme}><option value="system">{tr("跟随系统", "System")}</option><option value="light">{tr("浅色", "Light")}</option><option value="dark">{tr("深色", "Dark")}</option></select>}
          />
        </TMPanel>
      </div>


      <div className="settings-page-actions">
        <button onClick={resetSettingsDraft} type="button">{tr("恢复默认", "Reset")}</button>
        <button className="primary" onClick={saveSettings} type="button">{tr("应用设置", "Apply settings")}</button>
      </div>
    </div>
  );
}
