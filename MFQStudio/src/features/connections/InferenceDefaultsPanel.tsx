/** Edit immediate inference defaults without imposing arbitrary model output limits. */
import { localized } from '../../i18n/messages';
import { useTranslation } from 'react-i18next';
import { useRuntime } from '../../app/RuntimeProvider';
import { useSettings } from '../settings/SettingsProvider';
import { isValidMaxTokens, modeTemplateSettings, type GenerationSettings } from '../settings/configuration';
import { toast } from '../../stores/toastStore';
import { SectionLabel, SettingRow, TMPanel } from '../../app/display';
import { formatNumber } from '../../app/formatters';
/** Apply validated inference defaults to subsequent chat requests. */
export function InferenceDefaultsPanel() {
  const { settings, replaceSettings } = useSettings();
  const { t } = useTranslation();
  const { runtime, realtime } = useRuntime();
  function updateInference(patch: Partial<GenerationSettings>) {
    if (patch.maxTokens !== undefined && !isValidMaxTokens(patch.maxTokens)) {
      toast.error(localized('connections:inferenceDefaultsPanel.maximumOutputTokensMustBeAPositiveInteger'));
      return;
    }
    replaceSettings((current) => ({
      ...(current.inheritModelDefaults
        ? modeTemplateSettings(current, 'text', runtime, realtime)
        : current),
      ...patch,
      inheritModelDefaults: false,
      preset: 'custom',
    }));
  }
  return <>
        <SectionLabel title={t('connections:inferenceDefaultsPanel.chat')} />
        <TMPanel className="server-settings-panel">
          <div className="setting-list">
            <SettingRow
              title={t('connections:inferenceDefaultsPanel.systemPrompt')}
              detail={t('connections:inferenceDefaultsPanel.prependedToNewRequestsInTheBuiltInPlayground')}
              trailing={
                <textarea
                  aria-label={t('connections:inferenceDefaultsPanel.systemPrompt')}
                  className="server-prompt-input"
                  onChange={(event) => updateInference({ systemPrompt: event.target.value })}
                  placeholder={t('connections:inferenceDefaultsPanel.systemPrompt')}
                  rows={2}
                  value={settings.systemPrompt}
                />
              }
            />
            <SettingRow
              title={t('connections:inferenceDefaultsPanel.maximumOutput')}
              detail={t('connections:inferenceDefaultsPanel.completionLimitForTheBuiltInChat')}
              trailing={
                <div className="server-input-unit">
                  <input
                    aria-label={t('connections:inferenceDefaultsPanel.maximumOutput')}
                    className="server-number-input"
                    step={1}
                    min={1}
                    onChange={(event) => updateInference({ maxTokens: Number(event.target.value) })}
                    type="number"
                    value={settings.maxTokens}
                  />
                  <span>tokens</span>
                </div>
              }
            />
            <SettingRow
              title={t('connections:inferenceDefaultsPanel.temperature')}
              detail={t('connections:inferenceDefaultsPanel.samplingTemperatureBetween0And2')}
              trailing={
                <div className="server-temperature-control">
                  <input
                    aria-label={t('connections:inferenceDefaultsPanel.temperature')}
                    max={2}
                    min={0}
                    onChange={(event) =>
                      updateInference({ temperature: Number(event.target.value) })
                    }
                    step={0.05}
                    type="range"
                    value={settings.temperature}
                  />
                  <strong>{formatNumber(settings.temperature, 2)}</strong>
                </div>
              }
            />
          </div>
        </TMPanel>
  </>;
}
