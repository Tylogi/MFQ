import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router';
import { runtimeApi } from '../../shared/api/resources/runtime';
import { modelsApi } from '../../shared/api/resources/models';
import type { ModelArtifact, RuntimeProfile } from '../../shared/api/types';
import { useRuntime } from '../../app/RuntimeProvider';
import { useConnectionScope } from '../../app/useConnectionScope';
import { useSettings } from '../settings/SettingsProvider';
import { modeTemplateSettings } from '../settings/configuration';
import { Icon, SectionLabel, TMPanel } from '../../app/display';
import { errorMessage } from '../../app/formatters';
import { studioConfirm } from '../../studio';
import { STUDIO_PATHS } from '../../navigation';
import { toast } from '../../stores/toastStore';
import { ModelVendorMark } from '../../app/ModelVendorMark';

export function RuntimeProfilesPanel() {
  const { runtime, instances, realtime, ready, setSelectedModel, refreshRuntime } = useRuntime();
  const connectionScope = useConnectionScope();
  const { settings, tr, contextSize } = useSettings();
  const navigate = useNavigate();
  const [artifacts, setArtifacts] = useState<ModelArtifact[]>([]);
  const [runtimeProfiles, setRuntimeProfiles] = useState<RuntimeProfile[]>([]);
  const [profileName, setProfileName] = useState('');
  const [busy, setBusy] = useState(false);
  const currentInstance = instances.find((instance) => instance.id === runtime?.instance_id);
  const loadPinned = Boolean(currentInstance?.pinned);
  const loadIdleTtl = currentInstance?.idle_ttl_seconds ?? null;
  const resolvedGlobalSettings = settings.inheritModelDefaults
    ? modeTemplateSettings(settings, 'text', runtime, realtime)
    : settings;
  const thinkingSupported = runtime?.chat_template_capabilities?.thinking?.supported === true;
  useEffect(() => {
    if (!ready) return;
    let active = true;
    void Promise.all([modelsApi.modelArtifacts(), runtimeApi.runtimeProfiles()])
      .then(([nextArtifacts, profiles]) => {
        if (active) {
          setArtifacts(nextArtifacts);
          setRuntimeProfiles(profiles);
        }
      })
      .catch((cause) => {
        if (active) {
          toast.error(errorMessage(cause));
        }
      });
    return () => {
      active = false;
    };
  }, [ready, runtime?.model]);
  async function saveRuntimeProfile() {
    const current = connectionScope();
    const name = profileName.replace(/\s+/g, ' ').trim();
    const artifact = artifacts.find((item) => item.name === runtime?.model);
    if (busy || !name || !artifact) return;
    setBusy(true);
    try {
      await runtimeApi.createRuntimeProfile({
        name,
        load: {
          model: artifact.name,
          device_ids: [],
          idle_ttl_seconds: loadPinned ? null : loadIdleTtl,
          pin: loadPinned,
          context_size: currentInstance?.context_size ?? contextSize,
          prefill_chunk_size: 2048,
          sampling_defaults: {
            max_tokens: resolvedGlobalSettings.maxTokens,
            temperature: resolvedGlobalSettings.temperature,
            top_k: resolvedGlobalSettings.topK,
            top_p: resolvedGlobalSettings.topP,
            presence_penalty: resolvedGlobalSettings.presencePenalty,
            frequency_penalty: resolvedGlobalSettings.frequencyPenalty,
            repetition_penalty: resolvedGlobalSettings.repetitionPenalty,
            seed: resolvedGlobalSettings.seed,
            enable_thinking: thinkingSupported && resolvedGlobalSettings.enableThinking,
            enable_vision: resolvedGlobalSettings.enableVision,
            enable_mtp: resolvedGlobalSettings.enableMtp,
            reasoning_effort: resolvedGlobalSettings.reasoningEffort || null,
          },
        },
      });
      if (!current()) return;
      setProfileName('');
      const profiles = await runtimeApi.runtimeProfiles();
      if (!current()) return;
      setRuntimeProfiles(profiles);
      await refreshRuntime(false);
      if (!current()) return;
      toast.success(tr('运行配置已保存', 'Runtime profile saved'));
    } catch (cause) {
      if (current()) toast.error(errorMessage(cause));
    } finally {
      if (current()) setBusy(false);
    }
  }

  async function loadRuntimeProfile(profile: RuntimeProfile) {
    const current = connectionScope();
    if (busy) return;
    setBusy(true);
    try {
      if (profile.drifted && !(await studioConfirm(tr(
        '模型产物已变化。仍使用这个配置档案加载？',
        'The model artifact changed. Load this profile anyway?',
      ))) || !current()) return;
      await runtimeApi.loadRuntimeProfile(profile.id, profile.drifted);
      if (!current()) return;
      await refreshRuntime(false);
      if (!current()) return;
      setSelectedModel(profile.load.model);
      navigate(STUDIO_PATHS.models);
    } catch (cause) {
      if (current()) toast.error(errorMessage(cause));
    } finally {
      if (current()) setBusy(false);
    }
  }

  async function deleteRuntimeProfile(id: string) {
    const current = connectionScope();
    if (busy) return;
    setBusy(true);
    try {
      await runtimeApi.deleteRuntimeProfile(id);
      if (!current()) return;
      setRuntimeProfiles((current) => current.filter((item) => item.id !== id));
      toast.success(tr('运行配置已删除', 'Runtime profile deleted'));
    } catch (cause) {
      if (current()) toast.error(errorMessage(cause));
    } finally {
      if (current()) setBusy(false);
    }
  }

  return (
    <>
      <SectionLabel title={tr('运行配置', 'Runtime profiles')} />
      <TMPanel className="profile-panel">
        <div className="panel-heading">
          <div>
            <h2>{tr('已保存配置', 'Saved profiles')}</h2>
            <p>
              {tr(
                '将加载参数和采样默认值绑定到模型产物',
                'Bind load and sampling defaults to a model artifact',
              )}
            </p>
          </div>
          <b>{runtimeProfiles.length}</b>
        </div>
        <div className="profile-create">
          <input
            maxLength={64}
            onChange={(event) => setProfileName(event.target.value)}
            placeholder={tr('当前配置名称', 'Current configuration name')}
            value={profileName}
          />
          <button
            disabled={
              busy || !profileName.trim() || !artifacts.some((item) => item.name === runtime?.model)
            }
            onClick={() => void saveRuntimeProfile()}
            type="button"
          >
            {tr('保存当前配置', 'Save current')}
          </button>
        </div>
        {runtimeProfiles.length > 0 ? (
          <div className="profile-list">
            {runtimeProfiles.map((profile) => (
              <div className={`profile-row ${profile.drifted ? 'drifted' : ''}`} key={profile.id}>
                <div>
                  <strong className="model-identity-label">{profile.name}<ModelVendorMark name={profile.load.model} architecture={artifacts.find((item) => item.id === profile.artifact_id)?.architecture} size={20} /></strong>
                  <small>
                    {profile.load.context_size?.toLocaleString() ?? tr('自动', 'Auto')} ctx ·{' '}
                    {profile.load.prefill_chunk_size.toLocaleString()} chunk
                    {profile.drifted ? ` · ${tr('模型已变化', 'artifact changed')}` : ''}
                  </small>
                </div>
                <button
                  disabled={busy}
                  onClick={() => void loadRuntimeProfile(profile)}
                  type="button"
                >
                  {tr('加载', 'Load')}
                </button>
                <button
                  aria-label={tr('删除配置档案', 'Delete profile')}
                  disabled={busy}
                  onClick={() => void deleteRuntimeProfile(profile.id)}
                  type="button"
                >
                  <Icon name="trash" size={14} />
                </button>
              </div>
            ))}
          </div>
        ) : (
          <div className="inline-empty">
            {tr('尚未保存运行配置。', 'No runtime profiles saved yet.')}
          </div>
        )}
      </TMPanel>
    </>
  );
}
