/** 设置页面：展示推理草稿、采样参数、外观与数据导入导出操作。 */
import type { Dispatch, ReactNode, SetStateAction } from 'react';
import { SectionLabel, SettingRow, TMPanel } from '../../app/display';
import type { GenerationSettings, PresetName, UiLanguage, UiTheme } from './configuration';

export interface SettingsActions {
  /** 切换模型默认值继承，并更新待应用草稿。 */
  setModelDefaultInheritance: (enabled: boolean) => void;
  /** 将内置采样预设写入待应用草稿。 */
  applyPreset: (name: Exclude<PresetName, 'custom'>) => void;
  /** 使用当前上下文容量重载模型。 */
  reloadRuntime: () => Promise<void>;
  /** 导出本地偏好、预设和会话数据。 */
  exportStudioData: () => void;
  /** 导入备份数据并同步本地状态。 */
  importStudioData: (file: File) => Promise<void>;
  /** 导航到服务器配置页。 */
  openServerPage: () => void;
  /** 恢复草稿默认值，保留界面偏好。 */
  resetSettingsDraft: () => void;
  /** 应用当前设置草稿。 */
  saveSettings: () => void;
}
interface SettingsPageProps {
  tr: (zh: string, en: string) => string;
  settingsDraft: GenerationSettings;
  setSettingsDraft: Dispatch<SetStateAction<GenerationSettings>>;
  mtpAvailable: boolean;
  presetManager: ReactNode;
  contextCapacity?: number;
  contextSize: number;
  setContextSize: Dispatch<SetStateAction<number>>;
  busy: boolean;
  hasStudio: boolean;
  actions: SettingsActions;
}

/** 渲染可编辑设置草稿；保存、模型重载及数据操作由明确的业务回调执行。 */
export function SettingsPage({ tr, settingsDraft, setSettingsDraft, mtpAvailable, presetManager, contextCapacity, contextSize, setContextSize, busy, hasStudio, actions }: SettingsPageProps) {
  const { setModelDefaultInheritance, applyPreset, reloadRuntime, exportStudioData, importStudioData, openServerPage, resetSettingsDraft, saveSettings } = actions;
  return (
    <div className="settings-page">
      <SectionLabel title={tr("推理默认值", "Generation defaults")} />
      <TMPanel className="settings-page-panel settings-defaults-panel">
        <SettingRow
          title={tr("使用模型或架构默认值", "Use model or architecture defaults")}
          detail={tr(
            "优先读取模型元数据，缺失参数由模型型号或架构默认值补齐。",
            "Read model metadata first, then fill missing values from model or architecture defaults.",
          )}
          trailing={<input aria-label={tr("使用模型或架构默认值", "Use model or architecture defaults")} checked={settingsDraft.inheritModelDefaults} onChange={(event) => setModelDefaultInheritance(event.target.checked)} type="checkbox" />}
        />
        <SettingRow
          title={tr("视觉输入", "Vision input")}
          detail={tr("默认启用；关闭后图片和视频请求会被明确拒绝。", "On by default; when off, image and video requests are rejected explicitly.")}
          trailing={<input aria-label={tr("视觉输入", "Vision input")} checked={settingsDraft.enableVision} onChange={(event) => setSettingsDraft((current) => ({ ...current, enableVision: event.target.checked, inheritModelDefaults: false }))} type="checkbox" />}
        />
        <SettingRow
          title="MTP"
          detail={mtpAvailable
            ? tr("当前模型支持 MTP 投机解码。", "The current model supports MTP speculative decoding.")
            : tr("当前模型无法使用 MTP，将使用普通 Decode。", "MTP is unavailable for the current model; ordinary Decode will be used.")}
          trailing={<input aria-label="MTP" checked={mtpAvailable && settingsDraft.enableMtp} disabled={!mtpAvailable} onChange={(event) => setSettingsDraft((current) => ({ ...current, enableMtp: event.target.checked, inheritModelDefaults: false }))} type="checkbox" />}
        />
      </TMPanel>

      <fieldset className="settings-page-inherited" disabled={settingsDraft.inheritModelDefaults}>
        <div className="settings-page-grid">
          <div className="settings-page-section">
            <SectionLabel title={tr("预设与提示词", "Presets and prompt")} />
            <TMPanel className="settings-page-panel settings-form-panel">
              <div className="settings-control-block">
                <label>{tr("生成预设", "Generation preset")}</label>
                <div className="segmented">
                  {(["precise", "balanced", "creative"] as const).map((name) => (
                    <button aria-pressed={settingsDraft.preset === name} key={name} onClick={() => applyPreset(name)} type="button">
                      {name === "precise" ? tr("精确", "Precise") : name === "balanced" ? tr("均衡", "Balanced") : tr("创意", "Creative")}
                    </button>
                  ))}
                </div>
              </div>
              {presetManager}
              <div className="settings-control-block">
                <label htmlFor="settings-system-prompt">{tr("系统提示词", "System prompt")}</label>
                <textarea id="settings-system-prompt" onChange={(event) => setSettingsDraft((current) => ({ ...current, systemPrompt: event.target.value }))} rows={5} value={settingsDraft.systemPrompt} />
              </div>
              <SettingRow
                title={tr("排除历史思考", "Exclude reasoning history")}
                detail={tr("后续请求不再发送已保存的思考内容。", "Do not send saved reasoning in later requests.")}
                trailing={<input aria-label={tr("排除历史思考", "Exclude reasoning history")} checked={settingsDraft.excludeReasoning} onChange={(event) => setSettingsDraft((current) => ({ ...current, excludeReasoning: event.target.checked }))} type="checkbox" />}
              />
              <SettingRow
                title={tr("最大生成 token 数", "Maximum output tokens")}
                detail={tr("限制单次回答可生成的 token 数。", "Limit the number of tokens generated in one response.")}
                trailing={<input className="settings-number-input" max={65536} min={1} onChange={(event) => setSettingsDraft((current) => ({ ...current, maxTokens: Number(event.target.value) }))} type="number" value={settingsDraft.maxTokens} />}
              />
            </TMPanel>
          </div>

          <div className="settings-page-section">
            <SectionLabel title={tr("采样", "Sampling")} />
            <TMPanel className="settings-page-panel settings-form-panel settings-sampling-panel">
              {([
                [tr("温度", "Temperature"), "temperature", 0, 2, 0.05],
                [tr("核采样概率", "Top P"), "topP", 0.05, 1, 0.05],
                [tr("重复惩罚", "Repetition penalty"), "repetitionPenalty", 0.5, 2, 0.01],
                [tr("存在惩罚", "Presence penalty"), "presencePenalty", -2, 2, 0.05],
                [tr("频率惩罚", "Frequency penalty"), "frequencyPenalty", -2, 2, 0.05],
              ] as const).map(([label, key, min, max, step]) => (
                <label className="settings-range" key={key}>
                  <span>{label}<output>{settingsDraft[key].toFixed(2)}</output></span>
                  <input max={max} min={min} onChange={(event) => setSettingsDraft((current) => ({ ...current, [key]: Number(event.target.value), preset: "custom" }))} step={step} type="range" value={settingsDraft[key]} />
                </label>
              ))}
              <div className="settings-inline-fields">
                <label><span>{tr("候选词数", "Top K")}</span><input max={1024} min={0} onChange={(event) => setSettingsDraft((current) => ({ ...current, topK: Number(event.target.value), preset: "custom" }))} type="number" value={settingsDraft.topK} /></label>
                <label><span>{tr("随机种子", "Seed")}</span><input min={0} onChange={(event) => setSettingsDraft((current) => ({ ...current, seed: event.target.value ? Number(event.target.value) : null }))} placeholder={tr("随机", "Random")} type="number" value={settingsDraft.seed ?? ""} /></label>
              </div>
            </TMPanel>
          </div>
        </div>
      </fieldset>

      <div className="settings-page-grid">
        <div className="settings-page-section">
          <SectionLabel title={tr("上下文", "Context")} />
          <TMPanel className="settings-page-panel">
            <SettingRow
              title={tr("上下文窗口", "Context window")}
              detail={tr("更改后需要重新加载当前模型。", "Changing this value requires reloading the current model.")}
              trailing={<input className="settings-number-input settings-context-input" max={Number(contextCapacity) || 1048576} min={512} onChange={(event) => setContextSize(Number(event.target.value))} step={512} type="number" value={contextSize} />}
            />
            <div className="settings-panel-actions"><button className="secondary" disabled={busy} onClick={() => void reloadRuntime()} type="button">{tr("按此上下文重载模型", "Reload model with this context")}</button></div>
          </TMPanel>
        </div>

        <div className="settings-page-section">
          <SectionLabel title={tr("外观", "Appearance")} />
          <TMPanel className="settings-page-panel">
            <SettingRow
              title={tr("界面语言", "Interface language")}
              detail={tr("选择 MFQ Studio 的显示语言。", "Choose the display language for MFQ Studio.")}
              trailing={<select onChange={(event) => setSettingsDraft((current) => ({ ...current, language: event.target.value as UiLanguage }))} value={settingsDraft.language}><option value="system">{tr("跟随系统", "System")}</option><option value="zh-CN">简体中文</option><option value="en">English</option></select>}
            />
            <SettingRow
              title={tr("主题", "Theme")}
              detail={tr("跟随系统，或固定使用浅色或深色外观。", "Follow the system or use a fixed light or dark appearance.")}
              trailing={<select onChange={(event) => setSettingsDraft((current) => ({ ...current, theme: event.target.value as UiTheme }))} value={settingsDraft.theme}><option value="system">{tr("跟随系统", "System")}</option><option value="light">{tr("浅色", "Light")}</option><option value="dark">{tr("深色", "Dark")}</option></select>}
            />
          </TMPanel>
        </div>
      </div>

      <SectionLabel title={tr("数据与连接", "Data and connection")} />
      <TMPanel className="settings-page-panel settings-data-panel">
        <SettingRow
          title={tr("本地设置数据", "Local settings data")}
          detail={tr("导出或导入界面设置、生成预设和对话数据。", "Export or import interface settings, generation presets, and conversation data.")}
          trailing={<div className="portable-actions"><button onClick={exportStudioData} type="button">{tr("导出", "Export")}</button><label>{tr("导入", "Import")}<input accept="application/json,.json" onChange={(event) => { const file = event.target.files?.[0]; if (file) void importStudioData(file); event.target.value = ""; }} type="file" /></label></div>}
        />
        {hasStudio && <SettingRow title={tr("服务器连接", "Server connection")} detail={tr("配置本地或远程 MFQ Server。", "Configure a local or remote MFQ Server.")} trailing={<button className="secondary" onClick={openServerPage} type="button">{tr("打开服务器设置", "Open server settings")}</button>} />}
      </TMPanel>

      <div className="settings-page-actions">
        <button onClick={resetSettingsDraft} type="button">{tr("恢复默认", "Reset")}</button>
        <button className="primary" onClick={saveSettings} type="button">{tr("应用设置", "Apply settings")}</button>
      </div>
    </div>
  );
}
