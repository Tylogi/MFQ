/** 模型驻留策略的展示和表单控件。 */
import { useSettings } from '../settings/SettingsProvider';
import { TMPanel, SettingRow } from '../../app/display';
import { Switch } from '../../shared/ui/Switch';
import type { useModelCatalog } from './useModelCatalog';
import { ModelLoadContext } from './ModelLoadContext';

/** 展示内存固定和空闲卸载策略设置。 */
export function ModelLoadPolicy({ catalog }: { catalog: ReturnType<typeof useModelCatalog> }) {
  const { tr } = useSettings();
  const { loadPinned, setLoadPinned, loadIdleTtl, setLoadIdleTtl } = catalog;
  return (
    <TMPanel className="model-catalog-panel model-load-policy">
      <div className="panel-heading">
        <div>
          <h2>{tr('加载策略', 'Load policy')}</h2>
          <p>{tr('控制模型的驻留与自动卸载。', 'Control model residency and automatic unloading.')}</p>
        </div>
      </div>
      <div className="setting-list model-policy-panel">
        {!!catalog.runtime?.gpu_devices?.length && <SettingRow
          title={tr('计算设备', 'Compute device')}
          detail={tr('选择加载此模型的 GPU。', 'Choose the GPU for this model.')}
          trailing={<select aria-label={tr('计算设备', 'Compute device')}
            value={catalog.loadDeviceIds[0] ?? ''}
            onChange={event => catalog.setLoadDeviceIds(event.target.value ? [event.target.value] : [])}>
            <option value="">{tr('默认 GPU', 'Default GPU')}</option>
            {catalog.runtime.gpu_devices.map(device => <option key={device.id} value={device.id}>
              {`GPU ${device.id} · ${device.name}`}
            </option>)}
          </select>}
        />}
        <SettingRow
          title={tr('固定到内存', 'Pin in memory')}
          detail={tr('跳过 LRU 与空闲卸载', 'Skip LRU and idle eviction')}
          trailing={<Switch label={tr('固定到内存', 'Pin in memory')}
            checked={loadPinned} onCheckedChange={setLoadPinned} />}
        />
        <SettingRow
          title={tr('空闲卸载', 'Idle unload')}
          detail={loadPinned
            ? tr('固定模型不使用 TTL', 'Ignored while pinned')
            : tr('每次使用后重新计时', 'Resets after each use')}
          trailing={
            <select aria-label={tr('空闲卸载', 'Idle unload')} disabled={loadPinned}
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
      <ModelLoadContext key={catalog.connectionRevision} value={catalog.loadContext} onChange={catalog.setLoadContext}
        disabled={catalog.busy} models={catalog.artifacts} runtime={catalog.runtime} deviceIds={catalog.loadDeviceIds} />
    </TMPanel>
  );
}
