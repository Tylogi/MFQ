import { useState } from 'react';
import { useNavigate } from 'react-router';
import { sessionsApi } from '../../shared/api/resources/sessions';
import { presetsApi } from '../../shared/api/resources/presets';
import type { SessionArchive } from '../../shared/api/types';
import { ScreenHeader } from '../../app/display';
import { errorMessage } from '../../app/formatters';
import { STUDIO_PATHS } from '../../navigation';
import { ModelContextSettings } from '../runtime/ModelContextSettings';
import { useRuntime } from '../../app/RuntimeProvider';
import { useActiveSessionMode } from '../chat/hooks/useActiveSessionMode';
import { DEFAULT_SETTINGS, PRESETS, modeTemplateSettings, type PresetName } from './configuration';
import { presetResourceBody, storedPresetFromResource, type StoredPreset } from './presets';
import { SettingsPage } from './SettingsPage';
import { useSettings } from './SettingsProvider';
import { useGenerationPresets } from './useGenerationPresets';
import { UpdateManager, useStudioUpdateContext } from './UpdateManager';
import { toast } from '../../stores/toastStore';

export function SettingsRoute() {
  const { settings, replaceSettings, tr } = useSettings();
  const studioUpdates = useStudioUpdateContext();
  const {
    runtime,
    realtime,
    selectedModel,
    studio,
    ready,
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

  function setModelDefaultInheritance(enabled: boolean) {
    setDraft((current) => ({
      ...(enabled ? modeTemplateSettings(current, mode, runtime, realtime) : current),
      inheritModelDefaults: enabled,
    }));
    if (enabled) clearSelection();
  }
  function applyPreset(name: Exclude<PresetName, 'custom'>) {
    setDraft((current) => ({
      ...current,
      ...PRESETS[name],
      inheritModelDefaults: false,
      preset: name,
    }));
    clearSelection();
  }
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
        contextControls={<ModelContextSettings />}
        busy={busy}
        hasStudio={Boolean(studio)}
        updateManager={
          studio ? <UpdateManager {...studioUpdates} tr={tr} /> : undefined
        }
        actions={{
          setModelDefaultInheritance,
          applyPreset,
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
