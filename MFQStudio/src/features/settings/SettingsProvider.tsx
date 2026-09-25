/** 保存跨页面推理偏好和上下文容量，统一语言选择及主题持久化。 */
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type Dispatch,
  type ReactNode,
  type SetStateAction,
} from 'react';
import { loadSettings, SETTINGS_KEY, type GenerationSettings } from './configuration';

interface SettingsContextValue {
  settings: GenerationSettings;
  /** 合并已应用偏好，立即同步主题和本地存储。 */
  updateSettings: (patch: Partial<GenerationSettings>) => void;
  /** 用完整设置替换已应用偏好。 */
  replaceSettings: Dispatch<SetStateAction<GenerationSettings>>;
  tr: (zh: string, en: string) => string;
  english: boolean;
  contextSize: number;
  /** 更新模型加载和重载共用的上下文容量。 */
  setContextSize: Dispatch<SetStateAction<number>>;
}

const SettingsContext = createContext<SettingsContextValue | null>(null);

/** 在应用外壳提供共享偏好；草稿和业务请求由各页面自行持有。 */
export function SettingsProvider({ children }: { children: ReactNode }) {
  const [settings, replaceSettings] = useState(loadSettings);
  const [contextSize, setContextSize] = useState(32768);
  const english =
    settings.language === 'en' ||
    (settings.language === 'system' && !navigator.language.toLowerCase().startsWith('zh'));
  const tr = useCallback((zh: string, en: string) => (english ? en : zh), [english]);
  const updateSettings = useCallback(
    (patch: Partial<GenerationSettings>) =>
      replaceSettings((current) => ({ ...current, ...patch })),
    [],
  );

  useEffect(() => {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
    document.documentElement.dataset.theme = settings.theme;
  }, [settings]);

  const value = useMemo(
    () => ({ settings, replaceSettings, updateSettings, english, tr, contextSize, setContextSize }),
    [settings, updateSettings, english, tr, contextSize],
  );
  return <SettingsContext.Provider value={value}>{children}</SettingsContext.Provider>;
}

/** 读取应用共享设置，必须在 SettingsProvider 内调用。 */
export function useSettings() {
  const context = useContext(SettingsContext);
  if (!context) throw new Error('SettingsProvider is required');
  return context;
}
