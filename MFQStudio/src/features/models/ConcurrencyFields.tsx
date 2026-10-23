import { useState } from 'react';
import { SettingRow } from '../../app/display';
import type { RuntimeConcurrencyPolicy, RuntimeConcurrencySettings } from '../../shared/api/types';
import { useSettings } from '../settings/SettingsProvider';
import { DEFAULT_CONCURRENCY, validConcurrency, type ConcurrencyDefaults } from './concurrencySettings';

/** Shared load, runtime and preference controls with explicit inherited values. */
export function ConcurrencyFields({ value, onChange, policy, disabled = false }: {
  value: ConcurrencyDefaults;
  onChange(value: ConcurrencyDefaults): void;
  policy: RuntimeConcurrencyPolicy | null;
  disabled?: boolean;
}) {
  const { tr } = useSettings();
  const [drafts, setDrafts] = useState<Partial<Record<keyof RuntimeConcurrencySettings, string>>>({});
  const fields = [
    { key: 'max_requests', title: tr('同时请求数', 'Concurrent requests'),
      detail: tr('每多一个并发请求会多占用 KV 缓存；实际容量由模型回报，修改后需要重新加载。',
        'Each additional request uses more KV cache. The model reports actual capacity; '
        + 'changing rows requires a reload.') },
    { key: 'max_queued_requests', title: tr('队列上限', 'Queue limit'),
      detail: tr('队列满时拒绝新请求；0 表示不排队。', 'New requests are rejected when full; 0 disables waiting.') },
    { key: 'queue_timeout_seconds', title: tr('队列超时（秒）', 'Queue timeout (seconds)'),
      detail: tr('0 表示无限等待，仅在服务器未限制超时时可用。',
        '0 waits indefinitely, available when the server has no timeout ceiling.') },
  ] as const;
  const invalid = policy != null && !validConcurrency(value, policy);
  return <div className="setting-list">
    {fields.map(({ key, title, detail }) => {
      const current = drafts[key] ?? (value[key] == null ? '' : String(value[key]));
      const limit = policy?.limits[key];
      const fieldValid = policy == null || validConcurrency({ ...DEFAULT_CONCURRENCY, [key]: value[key] }, policy);
      return <SettingRow key={key} title={title} detail={`${detail} ${
        policy ? tr(`服务器上限：${limit ?? '无限制'}。留空沿用 ${policy.defaults[key]}。`,
          `Server limit: ${limit ?? 'unlimited'}. Leave empty to inherit ${policy.defaults[key]}.`)
          : tr('正在读取服务器上限…', 'Reading server limits…')}`}
        trailing={<input type="number" aria-label={title} aria-invalid={!fieldValid}
          min={key === 'max_requests' ? 1 : 0} max={limit ?? undefined}
          step={key === 'queue_timeout_seconds' ? 'any' : 1} disabled={disabled || !policy}
          placeholder={policy ? String(policy.defaults[key]) : '—'} value={current}
          onChange={event => {
            const text = event.target.value;
            setDrafts(previous => ({ ...previous, [key]: text }));
            onChange({ ...value, [key]: text === '' ? null : Number(text) });
          }} />} />;
    })}
    {invalid && <p role="alert">{tr('请输入服务器范围内的有效数值。', 'Enter valid values within the server limits.')}</p>}
  </div>;
}
