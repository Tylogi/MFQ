/** Display and edit model residency policies. */
import { useTranslation } from 'react-i18next';
import { TMPanel, SettingRow } from '../../app/display';
import { Switch } from '../../shared/ui/Switch';
import type { useModelCatalog } from './useModelCatalog';
/** Display memory-pinning and idle-unload policy settings. */
export function ModelLoadPolicy({ catalog }: { catalog: ReturnType<typeof useModelCatalog> }) {
  const { t } = useTranslation();
  const { loadPinned, setLoadPinned, loadIdleTtl, setLoadIdleTtl } = catalog;
  return (
    <TMPanel className="model-catalog-panel model-load-policy">
      <div className="panel-heading">
        <div>
          <h2>{t('models:modelLoadPolicy.loadPolicy')}</h2>
          <p>{t('models:modelLoadPolicy.controlModelResidencyAndAutomaticUnloading')}</p>
        </div>
      </div>
      <div className="setting-list model-policy-panel">
        <SettingRow
          title={t('models:modelLoadPolicy.pinInMemory')}
          detail={t('models:modelLoadPolicy.skipLruAndIdleEviction')}
          trailing={<Switch label={t('models:modelLoadPolicy.pinInMemory')}
            checked={loadPinned} onCheckedChange={setLoadPinned} />}
        />
        <SettingRow
          title={t('models:modelLoadPolicy.idleUnload')}
          detail={loadPinned
            ? t('models:modelLoadPolicy.ignoredWhilePinned')
            : t('models:modelLoadPolicy.resetsAfterEachUse')}
          trailing={
            <select aria-label={t('models:modelLoadPolicy.idleUnload')} disabled={loadPinned}
              onChange={(event) => setLoadIdleTtl(event.target.value ? Number(event.target.value) : null)}
              value={loadIdleTtl ?? ''}>
              <option value="">{t('models:modelLoadPolicy.never')}</option>
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
