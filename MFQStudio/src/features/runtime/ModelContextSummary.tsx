import { formatNumber } from '../../app/formatters';
import type { ModelCacheProfile } from '../../shared/api/types';
import type { KvQuantizationSettings } from '../../shared/api/resources/runtime';
import { useSettings } from '../settings/SettingsProvider';
import { contextCacheEstimate, streamingCacheEstimate } from '../models/cacheData';

interface Props {
  model: string;
  context: number;
  nativeContext?: number | null;
  yarnEnabled: boolean;
  profile?: ModelCacheProfile | null;
  streaming: boolean;
  budget: number;
  separateVram?: boolean;
  ramBudget?: number | null;
  changed: boolean;
  quantization?: KvQuantizationSettings;
}

export function ModelContextSummary({ model, context, nativeContext, yarnEnabled, profile, streaming, budget, separateVram = false, ramBudget, changed, quantization }: Props) {
  const { tr } = useSettings();
  const full = profile ? contextCacheEstimate(profile, context) : null;
  const stream = profile && streaming ? streamingCacheEstimate(profile, context, budget) : null;
  const capacity = (bytes: number | null | undefined) => bytes == null ? '—' : bytes >= 2 ** 30
    ? `${formatNumber(bytes / 2 ** 30, 2)} GiB` : `${formatNumber(bytes / 2 ** 20, 2)} MiB`;
  const factor = Number.isSafeInteger(context) && nativeContext ? Math.max(1, context / nativeContext).toFixed(2) : '—';
  return <output className="model-context-summary" aria-label={tr(`${model} 最终状态`, `${model} final state`)}>
    <div className="model-context-summary-heading"><strong>{tr('最终状态', 'Final state')}</strong>
      {changed && <span>{tr('未保存', 'Unsaved')}</span>}
    </div>
    <div className="model-context-summary-grid">
      <div><span>{tr('上下文', 'Context')}</span><b>{Number.isSafeInteger(context) && context >= 512 ? `${formatNumber(context)} tokens` : '—'}</b></div>
      <div><span>YaRN</span><b>{yarnEnabled ? `${factor}×` : tr('未启用 · 1.00×', 'Off · 1.00×')}</b></div>
      <div><span>Indexer</span><b>{capacity(full?.indexer)}</b>
        {streaming && <small><span>{separateVram ? tr('显存上限', 'VRAM limit') : tr('常驻', 'Resident')} {capacity(stream?.indexerLimit)}</span>{' · '}<span>{separateVram ? 'RAM / SSD' : 'SSD'} {capacity(stream?.indexerReadPerToken)}</span></small>}
      </div>
      <div><span>{tr('原始 KV', 'Raw KV')}</span><b>{capacity(full?.raw)}</b>
        {streaming && <small><span>{separateVram ? tr('显存预算', 'VRAM budget') : tr('常驻', 'Resident')} {capacity(stream?.rawFloor)}</span>{' · '}<span>{separateVram ? 'RAM / SSD' : 'SSD'} {capacity(full && stream ? Math.max(0, full.raw - stream.rawFloor) : null)}</span></small>}
      </div>
    </div>
    {streaming && separateVram && <div className="model-context-summary-reads">
      <span>{tr('KV 分层预算', 'KV tier budgets')}</span>
      <b>VRAM {capacity(budget)} · RAM {ramBudget == null ? tr('自动', 'Automatic') : capacity(ramBudget)} → SSD</b>
    </div>}
    <div className="model-context-summary-reads"><span>{tr('KV量化', 'KV quantization')}</span>
      <b>{quantization?.enabled ? `TurboQuant · K ${Math.floor(quantization.bits)} / V ${Math.ceil(quantization.bits)} bit` : tr('未启用', 'Off')}</b>
    </div>
    {streaming && <div className="model-context-summary-reads" title={tr('全部 QSA 层合计的最坏情况 SSD 读取量，不是实测流量；原始 KV 按所有检索位置均未命中计算。', 'Worst-case SSD reads across all QSA layers, not measured traffic; raw KV assumes every selected position misses the resident cache.')}>
      <span>{tr('最大读取量/token', 'Maximum reads/token')}</span>
      <span>Indexer <b>{capacity(stream?.indexerReadPerToken)}</b></span>
      <span>{tr('原始 KV', 'Raw KV')} <b>{capacity(stream?.rawReadPerToken)}</b></span>
    </div>}
  </output>;
}
