/** 设置路由独立管理编辑草稿、生成预设和备份导入导出。 */
import { useState } from 'react';
import { useNavigate } from 'react-router';
import { runtimeApi } from '../../shared/api/resources/runtime';
import { sessionsApi } from '../../shared/api/resources/sessions';
import { presetsApi } from '../../shared/api/resources/presets';
import type { SessionArchive } from '../../shared/api/types';
import { ScreenHeader } from '../../app/display';
import { errorMessage, formatNumber } from '../../app/formatters';
import { STUDIO_PATHS } from '../../navigation';
import { studioConfirm } from '../../studio';
import { useRuntime } from '../../app/RuntimeProvider';
import { useActiveSessionMode } from '../chat/hooks/useActiveSessionMode';
import { DEFAULT_SETTINGS, PRESETS, modeTemplateSettings, type PresetName } from './configuration';
import { presetResourceBody, storedPresetFromResource, type StoredPreset } from './presets';
import { SettingsPage } from './SettingsPage';
import { useSettings } from './SettingsProvider';
import { useGenerationPresets } from './useGenerationPresets';
import { toast } from '../../stores/toastStore';

/** 在访问设置路由时创建草稿并加载本领域数据，应用后才更新共享偏好。 */
export function SettingsRoute() {
  const { settings, replaceSettings, tr, contextSize, setContextSize } = useSettings();
  const {
    runtime,
    realtime,
    selectedModel,
    studio,
    ready,
    refreshRuntime,
    instances,
    capabilities,
  } = useRuntime();
  const mode = useActiveSessionMode();
  const selectedInstance = instances.find(
    (instance) => instance.model === selectedModel && instance.state !== 'failed',
  );
  const mtpAvailable =
    selectedInstance?.mtp_available ??
    (capabilities?.model === selectedModel && capabilities?.mtp_available === true);
  const navigate = useNavigate();
  const [draft, setDraft] = useState(() =>
    settings.inheritModelDefaults
      ? modeTemplateSettings(settings, mode, runtime, realtime)
      : settings,
  );
  const [busy, setBusy] = useState(false);
  const { presets, setPresets, clearSelection, manager } = useGenerationPresets(
    draft,
    setDraft,
    selectedModel,
    mode,
    ready,
  );

  /** 切换模型默认值继承，并清除不再匹配的预设选择。 */
  function setModelDefaultInheritance(enabled: boolean) {
    setDraft((current) => ({
      ...(enabled ? modeTemplateSettings(current, mode, runtime, realtime) : current),
      inheritModelDefaults: enabled,
    }));
    if (enabled) clearSelection();
  }
  /** 把内置参数组合写入草稿，不立即改变正在使用的设置。 */
  function applyPreset(name: Exclude<PresetName, 'custom'>) {
    setDraft((current) => ({
      ...current,
      ...PRESETS[name],
      inheritModelDefaults: false,
      preset: name,
    }));
    clearSelection();
  }
  /** 确认上下文变更后重载模型并刷新共享状态。 */
  async function reloadRuntime() {
    if (
      busy ||
      !(await studioConfirm(
        tr(
          `以 ${formatNumber(contextSize)} token 上下文重载模型？`,
          `Reload the model with a ${formatNumber(contextSize)} token context?`,
        ),
      ))
    )
      return;
    setBusy(true);
    try {
      await runtimeApi.reloadRuntime(contextSize, runtime?.instance_id);
      await refreshRuntime(true);
    } catch (cause) {
      toast.error(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }
  /** 按需读取会话并导出完整备份，及时释放下载对象 URL。 */
  async function exportStudioData() {
    setBusy(true);
    try {
      const sessions = await sessionsApi.listSessions();
      const archives = await Promise.all(sessions.map((session) => sessionsApi.exportSession(session.id)));
      const payload = {
        format: 'mfq-studio-export-v2',
        exported_at: new Date().toISOString(),
        sessions: archives,
        presets,
      };
      const anchor = document.createElement('a');
      const url = URL.createObjectURL(
        new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' }),
      );
      anchor.href = url;
      anchor.download = `mfq-studio-${new Date().toISOString().slice(0, 10)}.json`;
      try {
        anchor.click();
      } finally {
        URL.revokeObjectURL(url);
      }
    } catch (cause) {
      toast.error(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }
  /** 导入预设和会话归档，完成后通知聊天模块重读会话列表。 */
  async function importStudioData(file: File) {
    setBusy(true);
    try {
      const payload = JSON.parse(await file.text()) as {
        format?: string;
        presets?: StoredPreset[];
        sessions?: SessionArchive[];
      };
      if (
        payload.format !== 'mfq-studio-export-v2' ||
        !Array.isArray(payload.presets) ||
        !Array.isArray(payload.sessions)
      )
        throw new Error(tr('不是有效的 MFQ Studio 导出文件。', 'Not a valid MFQ Studio export.'));
      for (const preset of payload.presets) {
        if (!preset?.name || !preset.settings || !Number.isFinite(preset.contextSize)) continue;
        const existing = presets.find((item) => item.name === preset.name);
        const body = presetResourceBody(
          {
            ...preset,
            id: existing?.id,
            inheritGlobalSettings: preset.inheritGlobalSettings !== false,
          },
          selectedModel,
          mode,
        );
        if (existing?.id) await presetsApi.updateGenerationPreset(existing.id, body);
        else await presetsApi.createGenerationPreset(body);
      }
      for (const archive of payload.sessions) await sessionsApi.importSession(archive);
      setPresets((await presetsApi.generationPresets()).map(storedPresetFromResource));
      window.dispatchEvent(new Event('mfq:sessions-imported'));
      toast.success(tr('设置与会话导入成功', 'Settings and sessions imported successfully'));
    } catch (cause) {
      toast.error(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }
  /** 恢复模型默认参数，同时保留草稿内的外观偏好。 */
  function resetSettingsDraft() {
    setDraft({
      ...modeTemplateSettings(
        {
          ...DEFAULT_SETTINGS,
          language: draft.language,
          theme: draft.theme,
          playbackEnabled: draft.playbackEnabled,
        },
        mode,
        runtime,
        realtime,
      ),
      inheritModelDefaults: true,
    });
  }
  /** 将用户确认的草稿应用到所有后续请求。 */
  function saveSettings() {
    replaceSettings(draft);
    toast.success(tr('设置已应用', 'Settings applied successfully'));
  }
  return (
    <section className="dashboard-view">
      <ScreenHeader
        title={tr('设置', 'Settings')}
        subtitle={tr('推理默认值、外观与数据。', 'Generation defaults, appearance, and data.')}
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
        mtpAvailable={mtpAvailable}
        presetManager={manager}
        contextCapacity={runtime?.context_capacity}
        contextSize={contextSize}
        setContextSize={setContextSize}
        busy={busy}
        hasStudio={Boolean(studio)}
        actions={{
          setModelDefaultInheritance,
          applyPreset,
          reloadRuntime,
          exportStudioData,
          importStudioData,
          openServerPage: () => navigate(STUDIO_PATHS.server),
          resetSettingsDraft,
          saveSettings,
        }}
      />
    </section>
  );
}
