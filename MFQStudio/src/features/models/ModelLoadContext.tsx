import { useId, useState } from 'react';
import { CaretRightIcon } from '@phosphor-icons/react';
import { SettingRow } from '../../app/display';
import { useSettings } from '../settings/SettingsProvider';
import type { LoadContextDefaults } from './loadContextDefaults';
import type { KvQuantizationSettings } from '../../shared/api/resources/runtime';
import './model-load-context.css';

export function ModelLoadContext({ value, onChange, disabled = false }: {
  value: LoadContextDefaults; onChange: (value: LoadContextDefaults) => void; disabled?: boolean;
}) {
  const { tr } = useSettings();
  const id = useId();
  const [expanded, setExpanded] = useState(false);
  const [kind, setKind] = useState<'sparse' | 'full'>('sparse');
  const [contextDraft, setContextDraft] = useState<string | null>(null);
  const context = contextDraft ?? (value.context_size == null ? '' : String(value.context_size));
  const valid = context === '' || Number.isSafeInteger(Number(context)) && Number(context) >= 512 && Number(context) <= 2147483647;
  const patch = (change: Partial<LoadContextDefaults>) => onChange({ ...value, ...change });
  const locked = disabled || value.automatic;
  const sparse = kind === 'sparse';
  return <section className="model-load-context" aria-label={tr('上下文管理', 'Context management')}>
    <h3>{tr('上下文管理', 'Context management')}</h3>
    <SettingRow title={tr('默认上下文', 'Default context')}
      detail={tr('留空使用 32768 tokens；超出模型支持的上限时自动缩小。设置会保存在当前浏览器，下次加载模型时生效。',
        'Leave empty for 32768 tokens, capped by the model’s supported context. Settings are saved in this browser and apply on the next model load.')}
      trailing={<input type="number" min={512} max={2147483647} step={1} placeholder="32768" value={context}
        aria-label={tr('默认上下文', 'Default context')} aria-invalid={!valid} disabled={disabled}
        onChange={event => {
          const text = event.target.value;
          setContextDraft(text);
          if (text === '' || Number.isSafeInteger(Number(text)) && Number(text) >= 512 && Number(text) <= 2147483647)
            patch({ context_size: text === '' ? null : Number(text) });
        }} onBlur={() => { if (!valid) setContextDraft(null); }} />} />
    {!valid && <p role="alert">{tr('请输入 512～2147483647 的整数，或留空使用 32768。', 'Enter an integer from 512 to 2147483647, or leave empty for 32768.')}</p>}
    <div className="model-load-advanced-heading">
      <button type="button" className="model-load-advanced-toggle" aria-expanded={expanded} aria-controls={id}
        onClick={() => setExpanded(open => !open)}>
        <CaretRightIcon size={15} aria-hidden="true" />{tr('高级', 'Advanced')}
      </button>
      <label><input type="checkbox" checked={value.automatic} disabled={disabled}
        onChange={event => { patch({ automatic: event.target.checked }); if (!event.target.checked) setExpanded(true); }} />{tr('自动', 'Automatic')}</label>
    </div>
    <div id={id} hidden={!expanded} className="model-load-advanced-body">
      <p>{sparse
        ? tr('自动：关闭 SSD KV、开启预分配、关闭 KV 量化。取消勾选后可手动调整。',
          'Automatic: SSD KV off, preallocation on, KV quantization off. Uncheck to customize.')
        : tr('自动：开启预分配、关闭 KV 量化。取消勾选后可手动调整。',
          'Automatic: preallocation on, KV quantization off. Uncheck to customize.')}</p>
      <div className="model-load-context-kinds" role="group" aria-label={tr('模型类型', 'Model type')}>
        <button type="button" aria-pressed={sparse} onClick={() => setKind('sparse')}>{tr('稀疏注意力模型', 'Sparse-attention models')}</button>
        <button type="button" aria-pressed={!sparse} onClick={() => setKind('full')}>{tr('非稀疏注意力模型', 'Other models')}</button>
      </div>
      {sparse && <SettingRow title="SSD KV"
        detail={tr('显存和内存容纳不下的冷 KV 写入 SSD；长上下文可能变慢，并占用磁盘空间。',
          'Cold KV spills to SSD when VRAM and RAM are full; this can slow long contexts and use disk space.')}
        trailing={<input type="checkbox" aria-label="SSD KV" disabled={locked}
          checked={!value.automatic && value.ssd_kv}
          onChange={event => patch({ ssd_kv: event.target.checked, ...(event.target.checked ? {} : { ram_preallocated: true }) })} />} />}
      <SettingRow title={sparse ? tr('RAM 预分配', 'RAM preallocation') : tr('VRAM 预分配', 'VRAM preallocation')}
        detail={sparse
          ? tr('为显存预算之外的 KV 预留内存。关闭时会同时开启 SSD KV，以便内存不足时继续增长；SSD 空间不足会明确报错。',
            'Reserve RAM for KV beyond the VRAM budget. Turning this off also enables SSD KV for further growth; insufficient disk space produces an error.')
          : tr('加载时为目标上下文预留 KV 显存，关闭后按需分配。',
            'Reserve KV VRAM for the target context at load time. When off, allocate on demand.')}
        trailing={<input type="checkbox" aria-label={sparse ? tr('RAM 预分配', 'RAM preallocation') : tr('VRAM 预分配', 'VRAM preallocation')}
          disabled={locked} checked={value.automatic || (sparse ? value.ram_preallocated : value.vram_preallocated)}
          onChange={event => patch(sparse ? { ram_preallocated: event.target.checked, ...(!event.target.checked ? { ssd_kv: true } : {}) }
            : { vram_preallocated: event.target.checked })} />} />
      <SettingRow title={tr('默认 KV 量化档位', 'Default KV quantization')}
        detail={tr('适用于支持 TurboQuant 的模型。低精度节省 KV 空间，可能影响输出质量；Indexer Cache 和线性注意力递推状态保持原精度。不支持的模型使用原精度。',
          'Applies to models supporting TurboQuant. Lower precision saves KV space and may affect quality. Indexer caches and recurrent states retain their precision; unsupported models use native precision.')}
        trailing={<select aria-label={tr('默认 KV 量化档位', 'Default KV quantization')} disabled={locked}
          value={value.automatic || !value.kv_quantization.enabled ? 'off' : String(value.kv_quantization.bits)}
          onChange={event => patch({ kv_quantization: { algorithm: 'turboquant', enabled: event.target.value !== 'off',
            bits: event.target.value === 'off' ? value.kv_quantization.bits : Number(event.target.value) as KvQuantizationSettings['bits'] } })}>
          <option value="off">{tr('关闭（原精度）', 'Off (native precision)')}</option>
          {[2, 2.5, 3, 3.5, 4, 6, 8].map(bits => <option key={bits} value={bits}>{bits} bit</option>)}
        </select>} />
    </div>
  </section>;
}
