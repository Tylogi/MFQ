/** Store cross-page inference preferences and context capacity, and centralize language and theme persistence. */
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
  /** Merge applied preferences and immediately synchronize the theme and local storage. */
  updateSettings: (patch: Partial<GenerationSettings>) => void;
  /** Replace applied preferences with a complete settings object. */
  replaceSettings: Dispatch<SetStateAction<GenerationSettings>>;
  tr: (zh: string, en: string) => string;
  english: boolean;
  contextSize: number;
  /** Update the context capacity shared by model loading and reloads. */
  setContextSize: Dispatch<SetStateAction<number>>;
}

const SettingsContext = createContext<SettingsContextValue | null>(null);

/** Provide shared preferences through the app shell; pages own their drafts and business requests. */
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

/** Read shared application settings; must be called within SettingsProvider. */
export function useSettings() {
  const context = useContext(SettingsContext);
  if (!context) throw new Error('SettingsProvider is required');
  return context;
}
