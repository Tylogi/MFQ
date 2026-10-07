import { useEffect, useRef, useState } from 'react';
import { quantizationApi } from '../../shared/api/resources/quantization';
import { useSettings } from '../settings/SettingsProvider';
import './source-loading.css';

export function SourceLoadingControl({ model, purpose, contextSize, layerwise, onChange }: {
  model: string; purpose: 'imatrix' | 'wt2'; contextSize: number; layerwise: boolean; onChange: (value: boolean) => void;
}) {
  const { tr } = useSettings();
  const [checking, setChecking] = useState(false);
  const [message, setMessage] = useState('');
  const version = useRef(0);
  useEffect(() => {
    version.current++;
    onChange(true); setMessage(''); setChecking(false);
    return () => { version.current++; };
  }, [model, purpose, contextSize, onChange]);
  async function change(value: boolean) {
    const current = ++version.current;
    setMessage('');
    if (value) { onChange(true); setChecking(false); return; }
    setChecking(true);
    try {
      const plan = await quantizationApi.loadingPlan(model, purpose, contextSize);
      if (current !== version.current) return;
      onChange(!plan.resident_allowed);
      const available = (plan.available_bytes / 2 ** 30).toFixed(1);
      const required = (plan.resident_required_bytes / 2 ** 30).toFixed(1);
      setMessage(plan.resident_allowed
        ? tr(`常驻预计需要 ${required} GiB，当前可用 ${available} GiB；启动时再次检查。`, `Residency needs an estimated ${required} GiB; ${available} GiB is available. Checked again at startup.`)
        : tr(`空余内存不足（可用 ${available} GiB，预计需要 ${required} GiB），已恢复逐层。`, `Insufficient free memory (${available} GiB available, ${required} GiB estimated); layerwise loading restored.`));
    } catch {
      if (current === version.current) {
        onChange(true);
        setMessage(tr('无法确认可用资源，已恢复逐层。', 'Available resources could not be verified; layerwise loading restored.'));
      }
    } finally { if (current === version.current) setChecking(false); }
  }
  return <div className="source-loading-control">
    <label><input type="checkbox" checked={layerwise} disabled={checking || !model} onChange={(event) => { void change(event.target.checked); }} />{tr('逐层', 'Layerwise')}</label>
    <small>{tr('对于较大的模型，推荐大多数设备勾选', 'For larger models, recommended on most devices')}</small>
    {checking && <p role="status">{tr('检查当前空余内存…', 'Checking current free memory…')}</p>}
    {message && <p role="status">{message}</p>}
  </div>;
}
