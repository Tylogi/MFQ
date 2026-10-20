import { useEffect, useState, type ReactNode } from 'react';
import { getCurrentWindow } from '@tauri-apps/api/window';
import type { UnlistenFn } from '@tauri-apps/api/event';
import { CopySimpleIcon, MinusIcon, SquareIcon, XIcon } from '@phosphor-icons/react';
import { useSettings } from '../features/settings/SettingsProvider';
import { studioCustomTitlebar } from '../shared/platform/studio';
import mfqMark from '../assets/mfq-mark.svg';
import './desktop-window.css';

/** 窗口控制独立于业务路由，登录、加载及渲染失败时仍可操作窗口。 */
export function DesktopWindowFrame({ children }: { children: ReactNode }) {
  const { tr, settings } = useSettings();
  const [enabled, setEnabled] = useState(false);
  const [maximized, setMaximized] = useState(false);
  const [focused, setFocused] = useState(true);

  useEffect(() => {
    let disposed = false;
    const listeners: UnlistenFn[] = [];
    const retain = (unlisten: UnlistenFn) => disposed ? unlisten() : listeners.push(unlisten);
    const refresh = async () => {
      const value = await getCurrentWindow().isMaximized();
      if (!disposed) setMaximized(value);
    };
    void (async () => {
      if (!await studioCustomTitlebar() || disposed) return;
      setEnabled(true);
      const native = getCurrentWindow();
      await refresh();
      const active = await native.isFocused();
      if (!disposed) setFocused(active);
      retain(await native.onResized(() => { void refresh().catch(console.error); }));
      retain(await native.onFocusChanged(({ payload }) => {
        if (!disposed) setFocused(payload);
      }));
    })().catch(console.error);
    return () => {
      disposed = true;
      listeners.forEach(unlisten => unlisten());
    };
  }, []);

  useEffect(() => {
    if (enabled) {
      void getCurrentWindow().setTheme(settings.theme === 'system' ? null : settings.theme).catch(console.error);
    }
  }, [enabled, settings.theme]);

  async function toggleMaximize() {
    const native = getCurrentWindow();
    await native.toggleMaximize();
    setMaximized(await native.isMaximized());
  }

  const maximizeLabel = maximized ? tr('还原窗口', 'Restore window') : tr('最大化', 'Maximize');
  return (
    <div className={enabled ? 'desktop-window' : 'desktop-window-browser'} data-window-focused={focused}>
      {enabled && <header key="titlebar" className="desktop-titlebar" aria-label={tr('窗口标题栏', 'Window title bar')}>
        <div className="desktop-titlebar-drag" data-tauri-drag-region>
          <img src={mfqMark} alt="" draggable={false} />
          <span>MFQ Studio</span>
        </div>
        <div className="desktop-window-controls">
          <button type="button" aria-label={tr('最小化', 'Minimize')} title={tr('最小化', 'Minimize')}
            onClick={() => { void getCurrentWindow().minimize().catch(console.error); }}>
            <MinusIcon size={15} aria-hidden="true" />
          </button>
          <button type="button" aria-label={maximizeLabel} title={maximizeLabel}
            onClick={() => { void toggleMaximize().catch(console.error); }}>
            {maximized ? <CopySimpleIcon size={13} aria-hidden="true" /> : <SquareIcon size={13} aria-hidden="true" />}
          </button>
          <button type="button" className="desktop-window-close" aria-label={tr('关闭窗口', 'Close window')}
            title={tr('关闭窗口', 'Close window')}
            onClick={() => { void getCurrentWindow().close().catch(console.error); }}>
            <XIcon size={15} aria-hidden="true" />
          </button>
        </div>
      </header>}
      <div key="content" className="desktop-window-content">{children}</div>
    </div>
  );
}
