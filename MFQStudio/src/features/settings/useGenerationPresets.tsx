/** 管理设置页的服务端生成预设、本地缓存和预设编辑状态。 */
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

/** 在设置页加载预设，并将选择、保存和删除操作限定在该页面生命周期内。 */
export function useGenerationPresets(
  draft: GenerationSettings,
  setDraft: Dispatch<SetStateAction<GenerationSettings>>,
  model: string,
  mode: SessionMode,
  ready: boolean,
) {
  const { tr, contextSize, setContextSize } = useSettings();
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

  /** 清除预设选择，供内置预设和模型默认值切换使用。 */
  function clearSelection() {
    setSelected('');
    setName('');
    setStatus(null);
  }
  /** 将保存的推理快照和上下文容量载入待应用草稿。 */
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
    setStatus({ error: false, text: tr('预设已载入。', 'Preset loaded.') });
  }
  /** 创建或覆盖当前预设，成功后更新本地缓存。 */
  async function save() {
    const normalized = name.replace(/\s+/g, ' ').trim().slice(0, 64);
    if (!normalized) {
      setStatus({ error: true, text: tr('请输入预设名称。', 'Enter a preset name.') });
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
      setStatus({ error: false, text: tr('预设已保存。', 'Preset saved.') });
    } catch (cause) {
      setStatus({ error: true, text: errorMessage(cause) });
    } finally {
      setBusy(false);
    }
  }
  /** 经确认删除选中的服务端预设及本地副本。 */
  async function remove() {
    if (
      busy ||
      !selected ||
      !(await studioConfirm(tr(`删除预设“${selected}”？`, `Delete preset “${selected}”?`)))
    )
      return;
    setBusy(true);
    try {
      const preset = presets.find((item) => item.name === selected);
      if (preset?.id) await presetsApi.deleteGenerationPreset(preset.id);
      setPresets((current) => current.filter((item) => item.name !== selected));
      clearSelection();
      setStatus({ error: false, text: tr('预设已删除。', 'Preset deleted.') });
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
        <span>{tr('已保存预设', 'Saved presets')}</span>
        <select disabled={disabled} onChange={(event) => load(event.target.value)} value={selected}>
          <option value="">
            {presets.length
              ? tr('选择预设…', 'Select a preset…')
              : tr('还没有保存的预设', 'No saved presets')}
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
          aria-label={tr('预设名称', 'Preset name')}
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
          placeholder={tr('预设名称', 'Preset name')}
          value={name}
        />
        <button disabled={disabled} onClick={() => void save()} type="button">
          {selected && name.trim() === selected ? tr('覆盖', 'Update') : tr('保存', 'Save')}
        </button>
        <button
          aria-label={tr('删除预设', 'Delete preset')}
          className="preset-delete"
          disabled={disabled || !selected}
          onClick={() => void remove()}
          title={tr('删除预设', 'Delete preset')}
          type="button"
        >
          <Icon name="trash" size={14} />
        </button>
      </div>
      {status && (
        <p className={status.error ? 'preset-status error' : 'preset-status'}>{status.text}</p>
      )}
      <small>
        {tr(
          '保存系统提示词、上下文和生成参数；不包含界面语言与播放开关。',
          'Stores the system prompt, context, and generation parameters; interface and playback preferences stay separate.',
        )}
      </small>
    </div>
  );
  return { presets, setPresets, clearSelection, manager };
}
