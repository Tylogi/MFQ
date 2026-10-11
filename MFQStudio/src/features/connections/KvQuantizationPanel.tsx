import { useId, useState } from 'react';
import { CaretRightIcon } from '@phosphor-icons/react';
import { useSettings } from '../settings/SettingsProvider';
import { CompactSelect } from '../../shared/ui/CompactSelect';
import type { KvQuantizationSettings } from '../../shared/api/resources/runtime';

interface Props {
  supported: boolean;
  disabled: boolean;
  settings: KvQuantizationSettings;
  onChange: (value: KvQuantizationSettings) => void;
}

export function KvQuantizationPanel({ supported, disabled, settings, onChange }: Props) {
  const { tr } = useSettings();
  const contentId = useId();
  const [expanded, setExpanded] = useState(false);
  return <section className="qsa-kv-settings kv-quantization-settings">
    <div className="qsa-kv-heading">
      <div className="qsa-kv-title">
        <button type="button" className="qsa-kv-caret" aria-expanded={expanded} aria-controls={contentId}
          aria-label={tr('KV量化', 'KV quantization')} onClick={() => setExpanded(value => !value)}>
          <CaretRightIcon size={14} aria-hidden="true" />
        </button>
        <strong>{tr('KV量化', 'KV quantization')}</strong>
      </div>
      <label className="qsa-kv-toggle">
        <input type="checkbox" aria-label={tr('启用KV量化', 'Enable KV quantization')} checked={settings.enabled && supported}
          disabled={!supported || disabled} onChange={event => { onChange({ ...settings, enabled: event.target.checked }); setExpanded(true); }} />
        <span>{tr('启用', 'Enable')}</span>
      </label>
    </div>
    <div id={contentId} className="qsa-kv-body" hidden={!expanded}>
      <p className="qsa-kv-description">{supported
        ? tr('使用 TurboQuant 压缩原始 K/V 的 RAM 与 SSD 占用。对于稀疏注意力而言，Indexer Cache 保持原精度；对于混合模型而言，线性注意力的递推状态不参与量化。',
          'TurboQuant reduces RAM and SSD storage for raw K/V. For sparse attention, the Indexer Cache retains its original precision; for hybrid models, linear-attention recurrent states are not quantized.')
        : tr('当前模型或计算后端暂不支持 KV 量化。', 'KV quantization is not currently supported by this model or backend.')}</p>
      <div className="qsa-kv-fields qsa-kv-model-settings-fields">
        <label className="qsa-kv-field"><span>{tr('量化档位', 'Quantization level')}</span>
          <CompactSelect aria-label={tr('KV量化档位', 'KV quantization level')} value={settings.bits} disabled={!supported || disabled}
            onChange={event => onChange({ ...settings, bits: Number(event.target.value) as KvQuantizationSettings['bits'] })}>
            {[2, 2.5, 3, 3.5, 4, 6, 8].map(bits => <option key={bits} value={bits}>{bits} bit</option>)}
          </CompactSelect>
        </label>
        <div className="qsa-kv-field"><span>{tr('量化方式', 'Method')}</span><div className="qsa-kv-context"><strong>TurboQuant</strong></div></div>
      </div>
      <p className="qsa-kv-description qsa-kv-note">{tr(`K ${Math.floor(settings.bits)} bit · V ${Math.ceil(settings.bits)} bit。低档位可能影响输出质量；保存到该模型后重载生效。`,
        `K ${Math.floor(settings.bits)} bit · V ${Math.ceil(settings.bits)} bit. Low precision may affect output quality; save to this model to reload and apply.`)}</p>
    </div>
  </section>;
}
