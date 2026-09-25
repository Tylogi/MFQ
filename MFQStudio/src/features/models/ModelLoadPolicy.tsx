/** 模型驻留策略的展示和表单控件。 */
import { useSettings } from '../settings/SettingsProvider';
import { TMPanel, SettingRow } from '../../app/display';
import type { useModelCatalog } from './useModelCatalog';

/** 展示内存固定和空闲卸载策略设置。 */
export function ModelLoadPolicy({ catalog }: { catalog: ReturnType<typeof useModelCatalog> }) {
  const { tr } = useSettings();
  const { loadPinned, setLoadPinned, loadIdleTtl, setLoadIdleTtl } = catalog;
  return (
    <TMPanel className="model-catalog-panel">
      <div className="panel-heading">
        <div>
          <h2>{tr('加载策略', 'Load policy')}</h2>
          <p>{tr('控制模型的驻留与自动卸载。', 'Control model residency and automatic unloading.')}</p>
        </div>
      </div>
      <div className="setting-list model-policy-panel">
        <SettingRow
          title={tr('固定到内存', 'Pin in memory')}
          detail={tr('跳过 LRU 与空闲卸载', 'Skip LRU and idle eviction')}
          trailing={<input aria-label={tr('固定到内存', 'Pin in memory')}
            checked={loadPinned} onChange={(event) => setLoadPinned(event.target.checked)} type="checkbox" />}
        />
        <SettingRow
          title={tr('空闲卸载', 'Idle unload')}
          detail={loadPinned
            ? tr('固定模型不使用 TTL', 'Ignored while pinned')
            : tr('每次使用后重新计时', 'Resets after each use')}
          trailing={
            <select disabled={loadPinned}
              onChange={(event) => setLoadIdleTtl(event.target.value ? Number(event.target.value) : null)}
              value={loadIdleTtl ?? ''}>
              <option value="">{tr('永不', 'Never')}</option>
              <option value="300">5 min</option>
              <option value="900">15 min</option>
              <option value="3600">1 h</option>
            </select>
          }
        />
      </div>
    </TMPanel>
  );
}
