import { useEffect, useState } from 'react';
import type { ModelArtifact, ModelCacheProfile, RuntimeStatus } from '../../shared/api/types';
import { modelsApi } from '../../shared/api/resources/models';
import { formatNumber } from '../../app/formatters';
import { useSettings } from '../settings/SettingsProvider';
import { availableLoadMemory, estimateLoadMemory } from './loadMemoryEstimate';
import type { LoadContextDefaults } from './loadContextDefaults';

function gib(bytes: number | null) {
  return bytes == null ? '—' : `${formatNumber(bytes / 2 ** 30, 1)} GiB`;
}

export function ModelLoadMemoryEstimate({ models, runtime, deviceIds, value, valid, disabled }: {
  models: ModelArtifact[]; runtime?: RuntimeStatus | null; deviceIds: string[];
  value: LoadContextDefaults; valid: boolean; disabled: boolean;
}) {
  const { tr } = useSettings();
  const [selectedId, setSelectedId] = useState('');
  const [data, setData] = useState<{ key: string; profile: ModelCacheProfile | null } | null>(null);
  const choices = models.filter(model => model.complete && model.loadable);
  const selected = choices.find(model => model.id === selectedId);
  const key = selected ? `${selected.id}:${selected.modified_at}:${selected.total_bytes}` : '';
  useEffect(() => {
    if (!selected) { setSelectedId(''); return; }
    const controller = new AbortController();
    let active = true;
    void modelsApi.modelCacheProfile(selected.id, controller.signal, true)
      .then(profile => { if (active) setData({ key, profile }); })
      .catch(() => { if (active) setData({ key, profile: null }); });
    return () => { active = false; controller.abort(); };
  }, [selected?.id, key]);
  const available = availableLoadMemory(runtime, deviceIds);
  const loading = !!selected && data?.key !== key;
  const estimate = selected && data?.key === key && data.profile && valid
    ? estimateLoadMemory(selected, data.profile, value) : null;
  const exceeded = estimate != null && available != null && estimate.total > available;
  return <div className="model-load-memory-estimate">
    <p>{tr('选择模型，比较预计占用与剩余推理预算。自动预算已考虑系统预留和可回收内存。',
      'Select a model to compare estimated usage with the remaining inference budget. The automatic budget accounts for system headroom and reclaimable memory.')}</p>
    <div className="model-load-estimate-row">
      <strong>{tr('预计总占用（显存 + 内存）', 'Estimated total (VRAM + RAM)')}</strong>
      <div className="model-load-estimate-values">
        <span className="model-load-memory-available">{tr('当前剩余', 'Available now')} <strong>{gib(available)}</strong></span>
        <output aria-label={tr('预计总占用', 'Estimated total memory')} aria-live="polite"
          className={exceeded ? 'model-load-memory-exceeded' : undefined}
          title={estimate ? tr(`权重与预留 ${gib(estimate.weights)}，上下文缓存与递推状态 ${gib(estimate.caches)}（${estimate.context.toLocaleString()} tokens）。实际运行占用会随工作区变化。`,
            `Weights and reserve ${gib(estimate.weights)}, context caches and recurrent state ${gib(estimate.caches)} (${estimate.context.toLocaleString()} tokens). Runtime usage varies with workspaces.`) : undefined}>
          {!selected ? '—' : loading ? tr('估算中…', 'Estimating…') : !valid ? '—'
            : estimate ? gib(estimate.total) : tr('暂无法估算', 'Estimate unavailable')}
        </output>
        <select aria-label={tr('估算模型', 'Model to estimate')} value={selected?.id ?? ''} disabled={disabled}
          onChange={event => setSelectedId(event.target.value)}>
          <option value="">{tr('无', 'None')}</option>
          {choices.map(model => <option key={model.id} value={model.id}>{model.name}</option>)}
        </select>
      </div>
    </div>
  </div>;
}
