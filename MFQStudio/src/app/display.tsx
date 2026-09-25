/** 提供 Studio 现有图标、页面标题、业务面板与指标展示组件。 */
import { ReactNode } from 'react';
import { formatNumber } from './formatters';

export type ModelMonogramState = "idle" | "loading" | "ready" | "failed";

export type IconName =
  | "activity"
  | "chat"
  | "chart"
  | "check"
  | "clock"
  | "copy"
  | "flask"
  | "download"
  | "edit"
  | "folder"
  | "gauge"
  | "info"
  | "image"
  | "lightbulb"
  | "link"
  | "memory"
  | "moon"
  | "menu"
  | "paperclip"
  | "play"
  | "plus"
  | "queue"
  | "refresh"
  | "reuse"
  | "send"
  | "server-rack"
  | "settings"
  | "stop"
  | "sun"
  | "sun-moon"
  | "text-forward"
  | "trash"
  | "upload"
  | "volume"
  | "volume-off"
  | "waveform";

/** 渲染现有界面的命名图标，统一尺寸及无障碍装饰属性。 */
export function Icon({ name, size = 16 }: { name: IconName; size?: number }) {
  return (
    <svg
      aria-hidden="true"
      className="ui-icon"
      fill="none"
      height={size}
      viewBox="0 0 24 24"
      width={size}
    >
      {name === "activity" && <><path d="M3 12h4l2.5-7 5 14 2.5-7h4" /></>}
      {name === "chat" && <><path d="M21 15a4 4 0 0 1-4 4H8l-5 3V7a4 4 0 0 1 4-4h10a4 4 0 0 1 4 4z" /></>}
      {name === "chart" && <><path d="M4 19V5M4 19h16" /><path d="m7 15 4-4 3 2 5-6" /></>}
      {name === "check" && <path d="m5 12.5 4.5 4.5L19 7.5" />}
      {name === "clock" && <><circle cx="12" cy="12" r="9" /><path d="M12 7v5l3.5 2" /></>}
      {name === "copy" && <><rect height="13" rx="2" width="11" x="8" y="8" /><path d="M16 8V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h3" /></>}
      {name === "flask" && <><path d="M9 3h6M10 3v6l-5.7 9.2A1.8 1.8 0 0 0 5.8 21h12.4a1.8 1.8 0 0 0 1.5-2.8L14 9V3" /><path d="M7.5 15h9" /></>}
      {name === "download" && <><path d="M12 3v12" /><path d="m7 10 5 5 5-5" /><path d="M5 21h14" /></>}
      {name === "edit" && <><path d="M4 20h4l11-11a2.8 2.8 0 0 0-4-4L4 16z" /><path d="m13.5 6.5 4 4" /></>}
      {name === "folder" && <path d="M3 6.5A2.5 2.5 0 0 1 5.5 4H10l2 2h6.5A2.5 2.5 0 0 1 21 8.5v8A2.5 2.5 0 0 1 18.5 19h-13A2.5 2.5 0 0 1 3 16.5z" />}
      {name === "gauge" && <><path d="M4.2 18a8.5 8.5 0 1 1 15.6 0" /><path d="M6.5 15.5h.01M7.8 11h.01M12 8.8h.01M16.2 11h.01M17.5 15.5h.01" /><path d="m12 14 4.2-5.2" /><circle cx="12" cy="14" r="1.45" /></>}
      {name === "info" && <><circle cx="12" cy="12" r="9" /><path d="M12 10v6M12 7h.01" /></>}
      {name === "image" && <><rect height="16" rx="2" width="18" x="3" y="4" /><circle cx="8.5" cy="9" r="1.5" /><path d="m4.5 18 5-5 3 3 2.5-2.5 4.5 4.5" /></>}
      {name === "lightbulb" && <><path d="M9 18h6M10 22h4" /><path d="M8.4 14.7A6 6 0 1 1 15.6 14.7 4.1 4.1 0 0 0 14 18h-4a4.1 4.1 0 0 0-1.6-3.3z" /></>}
      {name === "link" && <><path d="m9.5 14.5 5-5" /><path d="M7.5 17.5 5 20a3.5 3.5 0 0 1-5-5l4-4a3.5 3.5 0 0 1 5 0" transform="translate(2 -2)" /><path d="m16.5 6.5 2.5-2.5a3.5 3.5 0 0 1 5 5l-4 4a3.5 3.5 0 0 1-5 0" transform="translate(-2 2)" /></>}
      {name === "memory" && <><rect height="14" rx="2" width="14" x="5" y="5" /><path d="M9 9h6v6H9zM9 2v3M15 2v3M9 19v3M15 19v3M2 9h3M2 15h3M19 9h3M19 15h3" /></>}
      {name === "moon" && <path d="M20.5 14.2A8.5 8.5 0 0 1 9.8 3.5 8.5 8.5 0 1 0 20.5 14.2z" />}
      {name === "menu" && <><path d="M4 7h16M4 12h16M4 17h16" /></>}
      {name === "paperclip" && <><path d="m20.5 11.5-8.8 8.8a6 6 0 0 1-8.5-8.5l9.5-9.5a4 4 0 0 1 5.7 5.7l-9.6 9.5a2 2 0 0 1-2.8-2.8l8.8-8.8" /></>}
      {name === "play" && <path d="m8 5 11 7-11 7z" fill="currentColor" stroke="none" />}
      {name === "plus" && <><path d="M12 5v14M5 12h14" /></>}
      {name === "queue" && <><path d="M8 7h12M8 12h12M8 17h12" /><circle cx="4" cy="7" r="1" fill="currentColor" stroke="none" /><circle cx="4" cy="12" r="1" fill="currentColor" stroke="none" /><circle cx="4" cy="17" r="1" fill="currentColor" stroke="none" /></>}
      {name === "refresh" && <><path d="M20 6v5h-5" /><path d="M4 18v-5h5" /><path d="M18.5 9A7 7 0 0 0 6.1 6.1L4 8M5.5 15A7 7 0 0 0 17.9 17.9L20 16" /></>}
      {name === "reuse" && <><path d="M20 7v5h-5M4 17v-5h5" /><path d="M18.3 9A7 7 0 0 0 6 6.5L4 9M5.7 15A7 7 0 0 0 18 17.5l2-2.5" /></>}
      {name === "send" && <><path d="m5 12 7-7 7 7M12 5v14" /></>}
      {name === "server-rack" && <><rect height="7" rx="1.8" width="16" x="4" y="3" /><rect height="7" rx="1.8" width="16" x="4" y="14" /><path d="M8 6.5h.01M8 17.5h.01M12 6.5h5M12 17.5h5" /></>}
      {name === "settings" && <><circle cx="12" cy="12" r="3" /><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.9l.1.1-2.8 2.8-.1-.1a1.7 1.7 0 0 0-1.9-.3 1.7 1.7 0 0 0-1 1.6v.2h-4V21a1.7 1.7 0 0 0-1-1.6 1.7 1.7 0 0 0-1.9.3l-.1.1L4.2 17l.1-.1a1.7 1.7 0 0 0 .3-1.9A1.7 1.7 0 0 0 3 14H2.8v-4H3a1.7 1.7 0 0 0 1.6-1 1.7 1.7 0 0 0-.3-1.9L4.2 7 7 4.2l.1.1A1.7 1.7 0 0 0 9 4.6 1.7 1.7 0 0 0 10 3v-.2h4V3a1.7 1.7 0 0 0 1 1.6 1.7 1.7 0 0 0 1.9-.3l.1-.1L19.8 7l-.1.1a1.7 1.7 0 0 0-.3 1.9 1.7 1.7 0 0 0 1.6 1h.2v4H21a1.7 1.7 0 0 0-1.6 1z" /></>}
      {name === "stop" && <><rect height="9" rx="1" width="9" x="7.5" y="7.5" /></>}
      {name === "sun" && <><circle cx="12" cy="12" r="3.5" /><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" /></>}
      {name === "sun-moon" && <><path d="M8.5 3.5A6.5 6.5 0 1 0 15 10a5 5 0 0 1-6.5-6.5z" /><path d="M17 3v2M17 9v2M13 7h2M19 7h2" /></>}
      {name === "text-forward" && <><path d="M4 6h10M4 10h8M4 14h6" /><path d="M13 17h7m-3-3 3 3-3 3" /></>}
      {name === "trash" && <><path d="M4 7h16M9 7V4h6v3M7 7l1 13h8l1-13M10 11v5M14 11v5" /></>}
      {name === "upload" && <><path d="M12 21V9" /><path d="m7 14 5-5 5 5" /><path d="M5 3h14" /></>}
      {name === "volume" && <><path d="M11 5 6.5 9H3v6h3.5L11 19z" /><path d="M15 9a4 4 0 0 1 0 6M18 6a8 8 0 0 1 0 12" /></>}
      {name === "volume-off" && <><path d="M11 5 6.5 9H3v6h3.5L11 19zM16 10l5 5M21 10l-5 5" /></>}
      {name === "waveform" && <path d="M3 12h2l2-6 3 12 3-12 2 6h6" />}
    </svg>
  );
}

/** 组合页面标题、副标题及可选页面操作。 */
export function ScreenHeader({
  title,
  subtitle,
  trailing,
}: {
  title: string;
  subtitle: string;
  trailing?: ReactNode;
}) {
  return (
    <header className="screen-header">
      <div>
        <h1>{title}</h1>
        <p>{subtitle}</p>
      </div>
      {trailing && <div className="screen-header-trailing">{trailing}</div>}
    </header>
  );
}

/** 展示业务分区标题及可选辅助文字。 */
export function SectionLabel({ title, subtitle }: { title: string; subtitle?: string }) {
  return (
    <div className="section-label">
      <strong>{title}</strong>
      {subtitle && <span>{subtitle}</span>}
    </div>
  );
}

/** 提供现有业务面板的统一容器样式。 */
export function TMPanel({
  children,
  className = "",
}: {
  children: ReactNode;
  className?: string;
}) {
  return <section className={`tm-panel ${className}`.trim()}>{children}</section>;
}

/** 根据模型名称及加载状态展示模型标记。 */
export function ModelMonogram({
  name,
  state,
}: {
  name: string;
  state: ModelMonogramState;
}) {
  const initial = Array.from(name.trim())[0]?.toLocaleUpperCase() || "E";
  return (
    <span aria-hidden="true" className={`model-monogram ${state}`}>
      {state === "idle" ? <img src="/mfq-mark.svg" alt="" /> : initial}
    </span>
  );
}

/** 展示单个运行指标及其解释信息。 */
export function MetricTile({
  label,
  value,
  detail,
  icon,
}: {
  label: string;
  value: string;
  detail: string;
  icon: IconName;
}) {
  return (
    <TMPanel className="metric-tile">
      <div className="metric-tile-label"><Icon name={icon} size={14} /><span>{label}</span></div>
      <strong>{value}</strong>
      <small>{detail}</small>
    </TMPanel>
  );
}

/** 排列设置项的标题、说明和尾部控件。 */
export function SettingRow({
  title,
  detail,
  trailing,
}: {
  title: string;
  detail: string;
  trailing: ReactNode;
}) {
  return (
    <div className="setting-row">
      <div><strong>{title}</strong><small>{detail}</small></div>
      <div className="setting-row-trailing">{trailing}</div>
    </div>
  );
}

/** 展示已用容量与总容量，并限制进度条范围。 */
export function UsageBar({
  label,
  used,
  total,
}: {
  label: string;
  used: number;
  total: number;
}) {
  const ratio = total > 0 ? Math.min(Math.max(used / total, 0), 1) : 0;
  return (
    <div className="usage-bar">
      <div><strong>{label}</strong><span>{formatNumber(used / 2 ** 30, 2)} / {formatNumber(total / 2 ** 30, 2)} GB</span></div>
      <div className="usage-bar-track"><i style={{ width: `${ratio * 100}%` }} /></div>
    </div>
  );
}

/** 呈现业务空状态及可选恢复操作。 */
export function EmptyPanel({
  icon,
  title,
  message,
  action,
}: {
  icon: IconName;
  title: string;
  message: string;
  action?: ReactNode;
}) {
  return (
    <TMPanel className="empty-panel">
      <Icon name={icon} size={30} />
      <strong>{title}</strong>
      <p>{message}</p>
      {action && <div className="empty-panel-action">{action}</div>}
    </TMPanel>
  );
}
