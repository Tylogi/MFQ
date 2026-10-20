import { useRuntime } from '../../app/RuntimeProvider';
import { SectionLabel, SettingRow, TMPanel } from '../../app/display';
import { formatNumber } from '../../app/formatters';
import { useSettings } from '../settings/SettingsProvider';
import { PrefixCacheControls } from './PrefixCacheControls';
import { hasSeparateVram } from '../runtime/memoryArchitecture';

export function PrefixCacheSettingsPanel() {
  const { runtime } = useRuntime();
  const { tr } = useSettings();
  const separate = hasSeparateVram(runtime);
  const diskBudget = Number(runtime?.prefix_cache_total_disk_max_bytes ?? runtime?.prefix_cache_disk_max_bytes ?? 0);
  return <>
    <SectionLabel title={tr('持久化前缀缓存', 'Persistent prefix cache')} />
    <TMPanel className="server-settings-panel server-prefix-panel">
      <div className="setting-list">
        <SettingRow title={tr('前缀缓存层级', 'Prefix cache hierarchy')}
          detail={separate ? tr('显存热层 + 低优先级 RAM 层 + 可选 SSD 冷层。', 'VRAM hot tier + low-priority RAM tier + optional SSD cold tier.') : tr('RAM 热层 + SSD 冷层。', 'RAM hot tier + SSD cold tier.')}
          trailing={<span className="server-managed-value">{separate ? 'VRAM → RAM → SSD' : 'RAM → SSD'}</span>} />
        <SettingRow
          title={tr('前缀块大小', 'Prefix block size')}
          detail={tr(
            '较大的块有利于长提示词吞吐，较小的块提供更细粒度的部分前缀复用。',
            'Larger blocks favor long-prompt throughput; smaller blocks allow finer partial-prefix reuse.',
          )}
          trailing={<div className="server-unit-value">
            <strong>{formatNumber(runtime?.prefix_cache_block_tokens || 256)}</strong>
            <span>tokens</span>
          </div>}
        />
        <PrefixCacheControls budget={diskBudget} />
      </div>
    </TMPanel>
  </>;
}
