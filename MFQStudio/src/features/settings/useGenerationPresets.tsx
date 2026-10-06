/** Manage server generation presets, local cache, and preset editing state for the settings page. */
import { useTranslation } from 'react-i18next';
import { useEffect, useState, type Dispatch, type SetStateAction } from 'react';
import { presetsApi } from '../../shared/api/resources/presets';
import type { SessionMode } from '../../shared/api/types';
import { Icon } from '../../app/display';
import { errorMessage } from '../../app/formatters';
import { studioConfirm } from '../../studio';
import type { GenerationSettings } from './configuration';
import {
  loadStoredPresets,
  presetResourceBody,
  presetSnapshot,
  STORED_PRESETS_KEY,
  storedPresetFromResource,
  type StoredPreset,
} from './presets';
import { useSettings } from './SettingsProvider';

/** Load presets on the settings page and scope selection, save, and delete operations to its lifecycle. */
export function useGenerationPresets(
  draft: GenerationSettings,
  setDraft: Dispatch<SetStateAction<GenerationSettings>>,
  model: string,
  mode: SessionMode,
  ready: boolean,
) {
  const { contextSize, setContextSize } = useSettings();
  const { t } = useTranslation();
  const [presets, setPresets] = useState(loadStoredPresets);
  const [selected, setSelected] = useState('');
  const [name, setName] = useState('');
  const [status, setStatus] = useState<{ error: boolean; text: string } | null>(null);
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    localStorage.setItem(STORED_PRESETS_KEY, JSON.stringify(presets));
  }, [presets]);
  useEffect(() => {
    if (!ready) return;
    let disposed = false;
    void presetsApi
      .generationPresets()
      .then((items) => {
        if (!disposed) setPresets(items.map(storedPresetFromResource));
      })
      .catch((cause) => {
        if (!disposed) setStatus({ error: true, text: errorMessage(cause) });
      });
    return () => {
      disposed = true;
    };
  }, [ready]);

  /** Clear the selected preset for built-in preset and model-default changes. */
  function clearSelection() {
    setSelected('');
    setName('');
    setStatus(null);
  }
  /** Load a saved inference snapshot and context capacity into the unapplied draft. */
  function load(name: string) {
    setSelected(name);
    setName(name);
    setStatus(null);
    const preset = presets.find((item) => item.name === name);
    if (!preset) return;
    setDraft((current) => ({
      ...current,
      ...preset.settings,
      inheritModelDefaults: false,
      preset: 'custom',
    }));
    setContextSize(preset.contextSize);
    setStatus({ error: false, text: t('settings:useGenerationPresets.presetLoaded') });
  }
  /** Create or overwrite the current preset and update the local cache on success. */
  async function save() {
    const normalized = name.replace(/\s+/g, ' ').trim().slice(0, 64);
    if (!normalized) {
      setStatus({ error: true, text: t('settings:useGenerationPresets.enterAPresetName') });
      return;
    }
    if (busy) return;
    setBusy(true);
    const existing = presets.find((item) => item.name === selected);
    const next: StoredPreset = {
      name: normalized,
      settings: presetSnapshot(draft),
      inheritGlobalSettings: false,
      contextSize,
      model,
      mode,
      updatedAt: new Date().toISOString(),
      icon: existing?.icon,
    };
    try {
      const body = presetResourceBody(next, model, mode);
      const resource = existing?.id
        ? await presetsApi.updateGenerationPreset(existing.id, body)
        : await presetsApi.createGenerationPreset(body);
      const saved = storedPresetFromResource(resource);
      setPresets((current) =>
        [...current.filter((item) => item.id !== saved.id && item.name !== selected), saved].slice(
          -50,
        ),
      );
      setSelected(saved.name);
      setName(saved.name);
      setStatus({ error: false, text: t('settings:useGenerationPresets.presetSaved') });
    } catch (cause) {
      setStatus({ error: true, text: errorMessage(cause) });
    } finally {
      setBusy(false);
    }
  }
  /** Delete the selected server preset and local copy after confirmation. */
  async function remove() {
    if (
      busy ||
      !selected ||
      !(await studioConfirm(t('settings:useGenerationPresets.deletePreset', { selected: selected })))
    )
      return;
    setBusy(true);
    try {
      const preset = presets.find((item) => item.name === selected);
      if (preset?.id) await presetsApi.deleteGenerationPreset(preset.id);
      setPresets((current) => current.filter((item) => item.name !== selected));
      clearSelection();
      setStatus({ error: false, text: t('settings:useGenerationPresets.presetDeleted') });
    } catch (cause) {
      setStatus({ error: true, text: errorMessage(cause) });
    } finally {
      setBusy(false);
    }
  }
  const disabled = draft.inheritModelDefaults || busy;
  const manager = (
    <div className="saved-presets">
      <label>
        <span>{t('settings:useGenerationPresets.savedPresets')}</span>
        <select disabled={disabled} onChange={(event) => load(event.target.value)} value={selected}>
          <option value="">
            {presets.length
              ? t('settings:useGenerationPresets.selectAPreset')
              : t('settings:useGenerationPresets.noSavedPresets')}
          </option>
          {presets.map((item) => (
            <option key={item.id ?? item.name} value={item.name}>
              {item.name}
            </option>
          ))}
        </select>
      </label>
      <div className="preset-save-row">
        <input
          aria-label={t('settings:useGenerationPresets.presetName')}
          disabled={disabled}
          maxLength={64}
          onChange={(event) => {
            setName(event.target.value);
            setStatus(null);
          }}
          onKeyDown={(event) => {
            if (event.key === 'Enter' && !event.nativeEvent.isComposing) {
              event.preventDefault();
              void save();
            }
          }}
          placeholder={t('settings:useGenerationPresets.presetName')}
          value={name}
        />
        <button disabled={disabled} onClick={() => void save()} type="button">
          {selected && name.trim() === selected ? t('settings:useGenerationPresets.update') : t('common:save')}
        </button>
        <button
          aria-label={t('settings:useGenerationPresets.deletePreset2')}
          className="preset-delete"
          disabled={disabled || !selected}
          onClick={() => void remove()}
          title={t('settings:useGenerationPresets.deletePreset2')}
          type="button"
        >
          <Icon name="trash" size={14} />
        </button>
      </div>
      {status && (
        <p className={status.error ? 'preset-status error' : 'preset-status'}>{status.text}</p>
      )}
      <small>
        {t('settings:useGenerationPresets.storesTheSystemPromptContextAndGenerationParametersInterfaceAndPlaybackPreferences')}
      </small>
    </div>
  );
  return { presets, setPresets, clearSelection, manager };
}
