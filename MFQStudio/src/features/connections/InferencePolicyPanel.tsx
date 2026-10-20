import { useEffect, useRef, useState } from 'react';
import { SectionLabel, SettingRow, TMPanel } from '../../app/display';
import { errorMessage } from '../../app/formatters';
import { useRuntime } from '../../app/RuntimeProvider';
import { runtimeApi } from '../../shared/api/resources/runtime';
import { Switch } from '../../shared/ui/Switch';
import { toast } from '../../stores/toastStore';
import { useSettings } from '../settings/SettingsProvider';

export function InferencePolicyPanel() {
  const { runtime, ready, refreshRuntime, connectionRevision } = useRuntime();
  const { settings, updateSettings, tr } = useSettings();
  const [busy, setBusy] = useState(false);
  const revision = useRef(connectionRevision);
  revision.current = connectionRevision;
  useEffect(() => { setBusy(false); }, [connectionRevision]);
  const enabled = runtime?.mtp_service_enabled;

  async function changeMtp(next: boolean) {
    if (busy || !ready || typeof enabled !== 'boolean') return;
    const currentRevision = connectionRevision;
    setBusy(true);
    try {
      await runtimeApi.configureInferencePolicy(next);
      if (revision.current !== currentRevision) return;
      await refreshRuntime();
      if (revision.current === currentRevision) toast.success(tr('服务 MTP 设置已保存', 'Service MTP setting saved'));
    } catch (cause) {
      if (revision.current === currentRevision) toast.error(errorMessage(cause, tr));
    } finally {
      if (revision.current === currentRevision) setBusy(false);
    }
  }

  return <>
    <SectionLabel title={tr('推理控制', 'Inference controls')} />
    <TMPanel>
      <SettingRow title="MTP"
        detail={tr('关闭后，所有模型与 API 请求均使用普通解码。切换无需重载模型。', 'When off, all models and API requests use ordinary decoding. No model reload is needed.')}
        trailing={<Switch label={tr('服务 MTP', 'Service MTP')} checked={enabled === true}
          disabled={!ready || busy || typeof enabled !== 'boolean'} onCheckedChange={(next) => void changeMtp(next)} />} />
      <SettingRow title={tr('视觉输入', 'Vision input')}
        detail={tr('默认启用；关闭后图片和视频请求会被明确拒绝。', 'On by default; when off, image and video requests are rejected explicitly.')}
        trailing={<Switch label={tr('视觉输入', 'Vision input')} checked={settings.enableVision}
          onCheckedChange={next => updateSettings({ enableVision: next, inheritModelDefaults: false })} />} />
    </TMPanel>
  </>;
}
