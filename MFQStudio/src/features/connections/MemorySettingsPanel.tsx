/** Provide MemorySettingsPanel interface behavior. */
import { useTranslation } from 'react-i18next';
import { useRuntime } from '../../app/RuntimeProvider';
import { SectionLabel, SettingRow, TMPanel } from '../../app/display';
import { formatBytes, formatNumber } from '../../app/formatters';
import { ModelContextSettings } from '../runtime/ModelContextSettings';
import { MemoryBudgetControls } from './MemoryBudgetControls';
import { PrefixCacheDirectory } from './PrefixCacheDirectory';
/** Compose runtime memory budgets, context limits, and prefix cache controls. */
export function MemorySettingsPanel() {
  const { runtime, instances } = useRuntime();
  const { t } = useTranslation();
  const loaded = instances.filter((item) => item.state === 'ready' || item.state === 'busy');
  const amounts = loaded.map((item) => item.memory?.resident_weight_bytes ?? item.resident_bytes);
  const memory = amounts.reduce<number>((sum, bytes) => sum + (bytes ?? 0), 0);
  const missing = amounts.some((bytes) => bytes == null);
  const residency = formatBytes(memory).replace(/\b(KB|MB|GB|TB)\b/g, (unit) => `${unit[0]}iB`);
  const diskBudget = Number(runtime?.prefix_cache_disk_max_bytes ?? 0);
  const persistent = typeof runtime?.prefix_cache_max_bytes === 'number';
  const hotOnly = runtime?.prefix_cache_mode === 'single_device_hot_prefix';
  return <>
        <SectionLabel title={t('connections:memorySettingsPanel.memoryPlan')} />
        <TMPanel className="server-settings-panel">
          <div className="setting-list">
            <MemoryBudgetControls residency={memory > 0 ? `${missing ? '≥ ' : ''}${residency}` : '--'} />
            <SettingRow
              title={t('connections:memorySettingsPanel.warmExpertCacheOnLaunch')}
              detail={t('connections:memorySettingsPanel.mfqWarmsAvailableExpertSlotsAccordingToTheArchitectureAndCurrentMemory')}
              trailing={<span className="server-managed-value">{t('connections:memorySettingsPanel.automatic')}</span>}
            />
            <SettingRow
              title={t('connections:memorySettingsPanel.prefixBlockSize')}
              detail={t('connections:memorySettingsPanel.largerBlocksFavorLongPromptThroughputSmallerBlocksAllowFinerPartialPrefix')}
              trailing={
                <div className="server-unit-value">
                  <strong>{formatNumber(runtime?.prefix_cache_block_tokens || 256)}</strong>
                  <span>tokens</span>
                </div>
              }
            />
            <ModelContextSettings />
          </div>
        </TMPanel>
        <SectionLabel title={t('connections:memorySettingsPanel.persistentPrefixCache')} />
        <TMPanel className="server-settings-panel">
          <div className="setting-list">
            <SettingRow
              title={t('connections:memorySettingsPanel.enableSsdTier')}
              detail={t('connections:memorySettingsPanel.persistIncrementalChecksummedPrefixBlocksAndRecoverThemAfterRestart')}
              trailing={
                <input
                  aria-label={t('connections:memorySettingsPanel.enableSsdTier')}
                  checked={persistent && !hotOnly}
                  disabled
                  readOnly
                  type="checkbox"
                />
              }
            />
            <PrefixCacheDirectory />
            <SettingRow
              title={t('connections:memorySettingsPanel.ssdBudget')}
              detail={t('connections:memorySettingsPanel.leafAwareLruKeepsDiskUsageWithinThisCeiling')}
              trailing={
                <div className="server-unit-value">
                  <strong>{diskBudget > 0 ? formatNumber(diskBudget / 2 ** 30, 0) : '--'}</strong>
                  <span>GiB</span>
                </div>
              }
            />
          </div>
        </TMPanel>
  </>;
}
