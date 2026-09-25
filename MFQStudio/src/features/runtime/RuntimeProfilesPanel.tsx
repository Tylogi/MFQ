/** 运行配置面板独立管理配置列表、名称草稿和保存加载操作。 */
import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router';
import { runtimeApi } from '../../shared/api/resources/runtime';
import { modelsApi } from '../../shared/api/resources/models';
import type { ModelArtifact, RuntimeProfile } from '../../shared/api/types';
import { useRuntime } from '../../app/RuntimeProvider';
import { useSettings } from '../settings/SettingsProvider';
import { modeTemplateSettings } from '../settings/configuration';
import { Icon, SectionLabel, TMPanel } from '../../app/display';
import { errorMessage } from '../../app/formatters';
import { studioConfirm } from '../../studio';
import { STUDIO_PATHS } from '../../navigation';
import { toast } from '../../stores/toastStore';

/** 按需读取模型与配置档案，保存当前实例策略及推理参数。 */
export function RuntimeProfilesPanel() {
  const { runtime, instances, realtime, ready, setSelectedModel, refreshRuntime } = useRuntime();
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
  /** 提交缓存或配置操作，保留确认语义并刷新对应资源。 */
  async function saveRuntimeProfile() {
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
          context_size: contextSize,
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
      setProfileName('');
      setRuntimeProfiles(await runtimeApi.runtimeProfiles());
      await refreshRuntime(false);
      toast.success(tr('运行配置已保存', 'Runtime profile saved'));
    } catch (cause) {
      toast.error(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  /** 提交缓存或配置操作，保留确认语义并刷新对应资源。 */
  async function loadRuntimeProfile(profile: RuntimeProfile) {
    if (busy) return;
    if (
      profile.drifted &&
      !(await studioConfirm(
        tr(
          '模型产物已变化。仍使用这个配置档案加载？',
          'The model artifact changed. Load this profile anyway?',
        ),
      ))
    )
      return;
    setBusy(true);
    try {
      await runtimeApi.loadRuntimeProfile(profile.id, profile.drifted);

      navigate(STUDIO_PATHS.models);
      await refreshRuntime(false);
      setSelectedModel(profile.load.model);
    } catch (cause) {
      toast.error(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  /** 提交缓存或配置操作，保留确认语义并刷新对应资源。 */
  async function deleteRuntimeProfile(id: string) {
    if (busy) return;
    setBusy(true);
    try {
      await runtimeApi.deleteRuntimeProfile(id);
      setRuntimeProfiles((current) => current.filter((item) => item.id !== id));
      toast.success(tr('运行配置已删除', 'Runtime profile deleted'));
    } catch (cause) {
      toast.error(errorMessage(cause));
    } finally {
      setBusy(false);
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
                  <strong>{profile.name}</strong>
                  <small>
                    {profile.load.context_size.toLocaleString()} ctx ·{' '}
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
