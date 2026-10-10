import { useRuntime } from '../../app/RuntimeProvider';
import { useSettings } from '../settings/SettingsProvider';
import { SectionLabel, SettingRow, TMPanel } from '../../app/display';
import { formatBytes } from '../../app/formatters';
import { MemoryBudgetControls } from './MemoryBudgetControls';
export function MemorySettingsPanel() {
  const { instances } = useRuntime();
  const { tr } = useSettings();
  const loaded = instances.filter((item) => item.state === 'ready' || item.state === 'busy');
  const amounts = loaded.map((item) => item.memory?.resident_weight_bytes ?? item.resident_bytes);
  const memory = amounts.reduce<number>((sum, bytes) => sum + (bytes ?? 0), 0);
  const missing = amounts.some((bytes) => bytes == null);
  const residency = formatBytes(memory).replace(/\b(KB|MB|GB|TB)\b/g, (unit) => `${unit[0]}iB`);
  return <>
        <SectionLabel title={tr('内存规划', 'Memory plan')} />
        <TMPanel className="server-settings-panel server-memory-panel">
          <div className="setting-list">
            <MemoryBudgetControls residency={memory > 0 ? `${missing ? '≥ ' : ''}${residency}` : '--'} />
            <SettingRow
              title={tr('启动时预热专家缓存', 'Warm expert cache on launch')}
              detail={tr(
                'MFQ 会依据架构和当前内存压力自动预热可用专家槽位。',
                'MFQ warms available expert slots according to the architecture and current memory pressure.',
              )}
              trailing={<span className="server-managed-value">{tr('自动', 'Automatic')}</span>}
            />
          </div>
        </TMPanel>
  </>;
}
