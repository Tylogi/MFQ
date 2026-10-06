/** Provide RuntimeProfilesPanel interface behavior. */
import { i18n } from '../../i18n';
import { localized } from '../../i18n/messages';
import { useTranslation } from 'react-i18next';
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
import { ModelVendorMark } from '../../app/ModelVendorMark';

/** Manage saved model loading profiles and apply them to the current runtime. */
export function RuntimeProfilesPanel() {
  const { runtime, instances, realtime, ready, setSelectedModel, refreshRuntime } = useRuntime();
  const { settings, contextSize } = useSettings();
  const { t } = useTranslation();
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
      toast.success(localized('runtime:runtimeProfilesPanel.runtimeProfileSaved'));
    } catch (cause) {
      toast.error(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  async function loadRuntimeProfile(profile: RuntimeProfile) {
    if (busy) return;
    if (
      profile.drifted &&
      !(await studioConfirm(
        t('runtime:runtimeProfilesPanel.theModelArtifactChangedLoadThisProfileAnyway'),
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

  async function deleteRuntimeProfile(id: string) {
    if (busy) return;
    setBusy(true);
    try {
      await runtimeApi.deleteRuntimeProfile(id);
      setRuntimeProfiles((current) => current.filter((item) => item.id !== id));
      toast.success(localized('runtime:runtimeProfilesPanel.runtimeProfileDeleted'));
    } catch (cause) {
      toast.error(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <SectionLabel title={t('runtime:runtimeProfilesPanel.runtimeProfiles')} />
      <TMPanel className="profile-panel">
        <div className="panel-heading">
          <div>
            <h2>{t('runtime:runtimeProfilesPanel.savedProfiles')}</h2>
            <p>
              {t('runtime:runtimeProfilesPanel.bindLoadAndSamplingDefaultsToAModelArtifact')}
            </p>
          </div>
          <b>{runtimeProfiles.length}</b>
        </div>
        <div className="profile-create">
          <input
            maxLength={64}
            onChange={(event) => setProfileName(event.target.value)}
            placeholder={t('runtime:runtimeProfilesPanel.currentConfigurationName')}
            value={profileName}
          />
          <button
            disabled={
              busy || !profileName.trim() || !artifacts.some((item) => item.name === runtime?.model)
            }
            onClick={() => void saveRuntimeProfile()}
            type="button"
          >
            {t('runtime:runtimeProfilesPanel.saveCurrent')}
          </button>
        </div>
        {runtimeProfiles.length > 0 ? (
          <div className="profile-list">
            {runtimeProfiles.map((profile) => (
              <div className={`profile-row ${profile.drifted ? 'drifted' : ''}`} key={profile.id}>
                <div>
                  <strong className="model-identity-label">{profile.name}<ModelVendorMark name={profile.load.model} architecture={artifacts.find((item) => item.id === profile.artifact_id)?.architecture} size={20} /></strong>
                  <small>
                    {profile.load.context_size.toLocaleString(i18n.resolvedLanguage)} ctx ·{' '}
                    {profile.load.prefill_chunk_size.toLocaleString(i18n.resolvedLanguage)} chunk
                    {profile.drifted ? ` · ${t('runtime:runtimeProfilesPanel.artifactChanged')}` : ''}
                  </small>
                </div>
                <button
                  disabled={busy}
                  onClick={() => void loadRuntimeProfile(profile)}
                  type="button"
                >
                  {t('runtime:runtimeProfilesPanel.load')}
                </button>
                <button
                  aria-label={t('runtime:runtimeProfilesPanel.deleteProfile')}
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
            {t('runtime:runtimeProfilesPanel.noRuntimeProfilesSavedYet')}
          </div>
        )}
      </TMPanel>
    </>
  );
}
