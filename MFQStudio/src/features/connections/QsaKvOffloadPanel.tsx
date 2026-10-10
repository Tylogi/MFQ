import { useId, useState } from 'react';
import { CaretRightIcon } from '@phosphor-icons/react';
import { formatNumber } from '../../app/formatters';
import { useSettings } from '../settings/SettingsProvider';
import type { ModelCacheProfile, RuntimeInstance } from '../../shared/api/types';
import { streamingCacheEstimate } from '../models/cacheData';

interface Props {
  model: RuntimeInstance;
  context: number;
  enabled: boolean;
  budget: string;
  profile?: ModelCacheProfile | null;
  metadataError?: string;
  disabled: boolean;
  onEnabledChange: (value: boolean) => void;
  onBudgetChange: (value: string) => void;
}

export function QsaKvOffloadPanel({ model, context, enabled, budget, profile, metadataError, disabled, onEnabledChange, onBudgetChange }: Props) {
  const { tr } = useSettings();
  const contentId = useId();
  const [expanded, setExpanded] = useState(false);
  const supported = model.qsa_kv_offload_supported === true;
  const budgetBytes = budget.trim() === '' ? 0 : Math.round(Number(budget) * 2 ** 30);
  const estimate = profile ? streamingCacheEstimate(profile, context, budgetBytes) : null;
  const capacity = (bytes: number) => bytes >= 2 ** 30 ? `${formatNumber(bytes / 2 ** 30, 2)} GiB` : `${formatNumber(bytes / 2 ** 20, 2)} MiB`;
  const estimateHint = estimate ? tr(`完整 Indexer 需求 ${capacity(estimate.requiredIndexer)}，按 KV 常驻预算限制上限；原始 KV 下界另扣除 I/O 缓冲。`,
    `Full Indexer requirement ${capacity(estimate.requiredIndexer)}, capped by the resident KV budget; the raw KV floor also subtracts I/O buffers.`)
    : metadataError || tr('模型缓存结构信息不可用或正在读取。', 'Model cache metadata is unavailable or loading.');
  return <section className="qsa-kv-settings">
    <div className="qsa-kv-heading">
      <div className="qsa-kv-title">
        <button type="button" className="qsa-kv-caret" aria-expanded={expanded} aria-controls={contentId}
          aria-label={tr('流式稀疏注意力', 'Streaming Sparse Attention')} onClick={() => setExpanded(value => !value)}>
          <CaretRightIcon size={14} aria-hidden="true" />
        </button>
        <strong>{tr('流式稀疏注意力', 'Streaming Sparse Attention')}</strong>
        <span>{tr('实验', 'Experimental')}</span>
      </div>
      <label className="qsa-kv-toggle">
        <input type="checkbox" aria-label={tr('启用流式稀疏注意力', 'Enable streaming sparse attention')}
          checked={enabled && supported} disabled={!supported || disabled}
          onChange={event => { onEnabledChange(event.target.checked); setExpanded(true); }} />
        <span>{tr('启用', 'Enable')}</span>
      </label>
    </div>
    <div id={contentId} className="qsa-kv-body" hidden={!expanded}>
      <p className="qsa-kv-description">{supported
        ? tr('允许稀疏注意力模型设置 KV Cache 内存驻留上限，大幅降低给定上下文下的驻留量，对推理速度影响较小。', 'Set a maximum resident-memory budget for sparse-attention KV cache, substantially reducing memory residency at a given context length with a small impact on inference speed.')
        : tr('仅支持具备此后端能力的 QSA 模型，其它模型不能启用。', 'Requires a QSA model and a backend with streaming sparse attention support.')}</p>
      <div className="qsa-kv-fields qsa-kv-model-settings-fields">
        <label className="qsa-kv-field" title={tr('按实际数据与 I/O 缓冲共享上限，不预留未来空间；块元数据与执行临时张量另计。', 'Shared payload and I/O budget, without future reservations; block metadata and execution scratch are separate.')}>
          <span>{tr('KV 常驻预算', 'Resident KV budget')}</span>
          <div className="qsa-kv-input"><input type="number" step="any" min="0" max={2 ** 20}
            aria-label={tr('KV 常驻预算', 'Resident KV budget')} value={budget} disabled={!supported || disabled}
            onChange={event => onBudgetChange(event.target.value)} /><span>GiB</span></div>
        </label>
        <div className="qsa-kv-field" title={tr('跟随该模型上方的上下文设置，保存时一并应用。', 'Follows the model context setting above and applies with the same save.')}>
          <span>{tr('最大上下文', 'Maximum context')}</span>
          <div className="qsa-kv-context" role="status" aria-label={tr('最大上下文', 'Maximum context')}>
            <strong>{Number.isSafeInteger(context) && context >= 512 ? formatNumber(context) : '—'}</strong><span>tokens</span>
          </div>
        </div>
      </div>
      {supported && <>
        <div className="qsa-kv-estimate" title={estimateHint}>
          <div><span>{tr('Indexer 上限估算', 'Indexer limit estimate')}</span><b>{estimate ? capacity(estimate.indexerLimit) : '—'}</b></div>
          <div><span>{tr('原始 KV 常驻下界', 'Raw KV residency floor')}</span><b>{estimate ? capacity(estimate.rawFloor) : '—'}</b></div>
          <div title={tr('全部 QSA 层合计的每 token 最大 SSD 读取量：完整 Indexer 需求减去常驻预算，放得下时为 0；非实测流量。', 'Maximum SSD reads per token across all QSA layers: full Indexer requirement minus the resident budget, or zero when it fits; not measured traffic.')}>
            <span>{tr('最大 Indexer 缓存读取量/token', 'Maximum Indexer reads/token')}</span><b>{estimate ? capacity(estimate.indexerReadPerToken) : '—'}</b>
          </div>
          <div title={tr('按模型的检索位置上限，假定原始 K、V 全部从 SSD 读取；为全部 QSA 层合计，非实测流量。', 'Assumes all selected raw K/V positions are read from SSD, using the model’s selection limit; summed across all QSA layers, not measured traffic.')}>
            <span>{tr('最大原始 KV 读取量/token', 'Maximum raw KV reads/token')}</span><b>{estimate?.rawReadPerToken != null ? capacity(estimate.rawReadPerToken) : '—'}</b>
          </div>
        </div>
        <p className="qsa-kv-description qsa-kv-note">{tr('预算较小时 Indexer 也会流式读取，延迟可能增加。', 'Small budgets also stream indexer keys, which may increase latency.')}</p>
      </>}
    </div>
  </section>;
}
