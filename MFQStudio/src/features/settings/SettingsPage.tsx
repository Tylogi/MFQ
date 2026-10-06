/** Render editable generation preferences, appearance controls, and archive actions. */
import type { TFunction } from 'i18next';
import type { Dispatch, ReactNode, SetStateAction } from 'react';
import { SectionLabel, SettingRow, TMPanel } from '../../app/display';
import { Switch } from '../../shared/ui/Switch';
import type { GenerationSettings, PresetName, UiLanguage, UiTheme } from './configuration';

export interface SettingsActions {
  setModelDefaultInheritance: (enabled: boolean) => void;
  applyPreset: (name: Exclude<PresetName, 'custom'>) => void;
  exportStudioData: () => void;
  importStudioData: (file: File) => Promise<void>;
  openServerPage: () => void;
  resetSettingsDraft: () => void;
  saveSettings: () => void;
}
interface SettingsPageProps {
  t: TFunction;
  settingsDraft: GenerationSettings;
  setSettingsDraft: Dispatch<SetStateAction<GenerationSettings>>;
  mtpAvailable: boolean;
  presetManager: ReactNode;
  contextControls?: ReactNode;
  busy: boolean;
  hasStudio: boolean;
  updateManager?: ReactNode;
  actions: SettingsActions;
}

/** Display settings drafts; the route validates and applies changes. */
export function SettingsPage({ t, settingsDraft, setSettingsDraft, mtpAvailable, presetManager, contextControls, busy, hasStudio, updateManager, actions }: SettingsPageProps) {
  const { setModelDefaultInheritance, applyPreset, exportStudioData, importStudioData, openServerPage, resetSettingsDraft, saveSettings } = actions;
  return (
    <div className="settings-page">
      <SectionLabel title={t('settings:settingsPage.generationDefaults')} />
      <TMPanel className="settings-page-panel settings-defaults-panel">
        <SettingRow
          title={t('settings:settingsPage.useModelOrArchitectureDefaults')}
          detail={t('settings:settingsPage.readModelMetadataFirstThenFillMissingValuesFromModelOrArchitecture')}
          trailing={<Switch label={t('settings:settingsPage.useModelOrArchitectureDefaults')} checked={settingsDraft.inheritModelDefaults} onCheckedChange={setModelDefaultInheritance} />}
        />
        <SettingRow
          title={t('settings:settingsPage.visionInput')}
          detail={t('settings:settingsPage.onByDefaultWhenOffImageAndVideoRequestsAreRejectedExplicitly')}
          trailing={<Switch label={t('settings:settingsPage.visionInput')} checked={settingsDraft.enableVision} onCheckedChange={(checked) => setSettingsDraft((current) => ({ ...current, enableVision: checked, inheritModelDefaults: false }))} />}
        />
        <SettingRow
          title="MTP"
          detail={mtpAvailable
            ? t('settings:settingsPage.theCurrentModelSupportsMtpSpeculativeDecoding')
            : t('settings:settingsPage.mtpIsUnavailableForTheCurrentModelOrdinaryDecodeWillBeUsed')}
          trailing={<Switch label="MTP" checked={mtpAvailable && settingsDraft.enableMtp} disabled={!mtpAvailable} onCheckedChange={(checked) => setSettingsDraft((current) => ({ ...current, enableMtp: checked, inheritModelDefaults: false }))} />}
        />
      </TMPanel>

      <fieldset className="settings-page-inherited" disabled={settingsDraft.inheritModelDefaults}>
        <div className="settings-page-grid">
          <div className="settings-page-section">
            <SectionLabel title={t('settings:settingsPage.presetsAndPrompt')} />
            <TMPanel className="settings-page-panel settings-form-panel">
              <div className="settings-control-block">
                <label>{t('settings:settingsPage.generationPreset')}</label>
                <div className="segmented">
                  {(["precise", "balanced", "creative"] as const).map((name) => (
                    <button aria-pressed={settingsDraft.preset === name} key={name} onClick={() => applyPreset(name)} type="button">
                      {name === "precise" ? t('settings:settingsPage.precise') : name === "balanced" ? t('settings:settingsPage.balanced') : t('settings:settingsPage.creative')}
                    </button>
                  ))}
                </div>
              </div>
              {presetManager}
              <div className="settings-control-block">
                <label htmlFor="settings-system-prompt">{t('settings:settingsPage.systemPrompt')}</label>
                <textarea id="settings-system-prompt" onChange={(event) => setSettingsDraft((current) => ({ ...current, systemPrompt: event.target.value }))} rows={5} value={settingsDraft.systemPrompt} />
              </div>
              <SettingRow
                title={t('settings:settingsPage.excludeReasoningHistory')}
                detail={t('settings:settingsPage.doNotSendSavedReasoningInLaterRequests')}
                trailing={<Switch label={t('settings:settingsPage.excludeReasoningHistory')} checked={settingsDraft.excludeReasoning} onCheckedChange={(checked) => setSettingsDraft((current) => ({ ...current, excludeReasoning: checked }))} />}
              />
              <SettingRow
                title={t('settings:settingsPage.maximumOutputTokens')}
                detail={t('settings:settingsPage.limitTheNumberOfTokensGeneratedInOneResponse')}
                trailing={<input aria-label={t('settings:settingsPage.maximumOutputTokens')} className="settings-number-input" min={1} step={1} onChange={(event) => setSettingsDraft((current) => ({ ...current, maxTokens: Number(event.target.value) }))} type="number" value={settingsDraft.maxTokens} />}
              />
            </TMPanel>
          </div>

          <div className="settings-page-section">
            <SectionLabel title={t('settings:settingsPage.sampling')} />
            <TMPanel className="settings-page-panel settings-form-panel settings-sampling-panel">
              {([
                [t('settings:settingsPage.temperature'), "temperature", 0, 2, 0.05],
                [t('settings:settingsPage.topP'), "topP", 0.05, 1, 0.05],
                [t('settings:settingsPage.repetitionPenalty'), "repetitionPenalty", 0.5, 2, 0.01],
                [t('settings:settingsPage.presencePenalty'), "presencePenalty", -2, 2, 0.05],
                [t('settings:settingsPage.frequencyPenalty'), "frequencyPenalty", -2, 2, 0.05],
              ] as const).map(([label, key, min, max, step]) => (
                <label className="settings-range" key={key}>
                  <span>{label}<output>{settingsDraft[key].toFixed(2)}</output></span>
                  <input max={max} min={min} onChange={(event) => setSettingsDraft((current) => ({ ...current, [key]: Number(event.target.value), preset: "custom" }))} step={step} type="range" value={settingsDraft[key]} />
                </label>
              ))}
              <div className="settings-inline-fields">
                <label><span>{t('settings:settingsPage.topK')}</span><input max={1024} min={0} onChange={(event) => setSettingsDraft((current) => ({ ...current, topK: Number(event.target.value), preset: "custom" }))} type="number" value={settingsDraft.topK} /></label>
                <label><span>{t('settings:settingsPage.seed')}</span><input min={0} onChange={(event) => setSettingsDraft((current) => ({ ...current, seed: event.target.value ? Number(event.target.value) : null }))} placeholder={t('settings:settingsPage.random')} type="number" value={settingsDraft.seed ?? ""} /></label>
              </div>
            </TMPanel>
          </div>
        </div>
      </fieldset>

      <div className="settings-page-grid">
        <div className="settings-page-section">
          <SectionLabel title={t('settings:settingsPage.context')} />
          <TMPanel className="settings-page-panel">
            {contextControls}
          </TMPanel>
        </div>

        <div className="settings-page-section">
          <SectionLabel title={t('settings:settingsPage.appearance')} />
          <TMPanel className="settings-page-panel">
            <SettingRow
              title={t('settings:settingsPage.interfaceLanguage')}
              detail={t('settings:settingsPage.chooseTheDisplayLanguageForMfqStudio')}
              trailing={<select aria-label={t('settings:settingsPage.interfaceLanguage')} onChange={(event) => setSettingsDraft((current) => ({ ...current, language: event.target.value as UiLanguage }))} value={settingsDraft.language}><option value="system">{t('settings:settingsPage.system')}</option><option value="zh-CN">简体中文</option><option value="en">English</option></select>}
            />
            <SettingRow
              title={t('settings:settingsPage.theme')}
              detail={t('settings:settingsPage.followTheSystemOrUseAFixedLightOrDarkAppearance')}
              trailing={<select aria-label={t('settings:settingsPage.theme')} onChange={(event) => setSettingsDraft((current) => ({ ...current, theme: event.target.value as UiTheme }))} value={settingsDraft.theme}><option value="system">{t('settings:settingsPage.system')}</option><option value="light">{t('settings:settingsPage.light')}</option><option value="dark">{t('settings:settingsPage.dark')}</option></select>}
            />
          </TMPanel>
        </div>
      </div>

      <SectionLabel title={t('settings:settingsPage.dataAndConnection')} />
      <TMPanel className="settings-page-panel settings-data-panel">
        <SettingRow
          title={t('settings:settingsPage.localSettingsData')}
          detail={t('settings:settingsPage.exportOrImportInterfaceSettingsGenerationPresetsAndConversationData')}
          trailing={<div className="portable-actions"><button onClick={exportStudioData} type="button">{t('settings:settingsPage.export')}</button><label>{t('settings:settingsPage.import')}<input accept="application/json,.json" onChange={(event) => { const file = event.target.files?.[0]; if (file) void importStudioData(file); event.target.value = ""; }} type="file" /></label></div>}
        />
        {hasStudio && <SettingRow title={t('settings:settingsPage.serverConnection')} detail={t('settings:settingsPage.configureALocalOrRemoteMfqServer')} trailing={<button className="secondary" onClick={openServerPage} type="button">{t('settings:settingsPage.openServerSettings')}</button>} />}
      </TMPanel>

      {updateManager && (
        <>
          <SectionLabel title={t('settings:settingsPage.version')} />
          <TMPanel className="settings-page-panel settings-update-panel">
            {updateManager}
          </TMPanel>
        </>
      )}

      <div className="settings-page-actions">
        <button onClick={resetSettingsDraft} type="button">{t('settings:settingsPage.reset')}</button>
        <button className="primary" onClick={saveSettings} type="button">{t('settings:settingsPage.applySettings')}</button>
      </div>
    </div>
  );
}
