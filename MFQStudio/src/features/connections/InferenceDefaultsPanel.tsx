/** 管理连接页的即时推理默认值，直接写入共享已应用设置。 */
import { useRuntime } from '../../app/RuntimeProvider';
import { useSettings } from '../settings/SettingsProvider';
import { modeTemplateSettings, type GenerationSettings } from '../settings/configuration';
import { SectionLabel, SettingRow, TMPanel } from '../../app/display';
import { formatNumber } from '../../app/formatters';
/** 修改默认提示词和采样参数时不依赖服务器配置草稿。 */
export function InferenceDefaultsPanel() {
  const { settings, replaceSettings, tr } = useSettings();
  const { runtime, realtime } = useRuntime();
  /** 保存推理字段前展开模型默认值，避免修改被继承模式覆盖。 */
  function updateInference(patch: Partial<GenerationSettings>) {
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
        <SectionLabel title={tr('对话', 'Chat')} />
        <TMPanel className="server-settings-panel">
          <div className="setting-list">
            <SettingRow
              title={tr('系统提示词', 'System prompt')}
              detail={tr(
                '添加到内置对话中新请求的开头。',
                'Prepended to new requests in the built-in playground.',
              )}
              trailing={
                <textarea
                  aria-label={tr('系统提示词', 'System prompt')}
                  className="server-prompt-input"
                  onChange={(event) => updateInference({ systemPrompt: event.target.value })}
                  placeholder={tr('系统提示词', 'System prompt')}
                  rows={2}
                  value={settings.systemPrompt}
                />
              }
            />
            <SettingRow
              title={tr('最大输出', 'Maximum output')}
              detail={tr('内置对话的最大生成长度。', 'Completion limit for the built-in chat.')}
              trailing={
                <div className="server-input-unit">
                  <input
                    aria-label={tr('最大输出', 'Maximum output')}
                    className="server-number-input"
                    max={65536}
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
              title={tr('温度', 'Temperature')}
              detail={tr('采样温度范围为 0 到 2。', 'Sampling temperature between 0 and 2.')}
              trailing={
                <div className="server-temperature-control">
                  <input
                    aria-label={tr('温度', 'Temperature')}
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
