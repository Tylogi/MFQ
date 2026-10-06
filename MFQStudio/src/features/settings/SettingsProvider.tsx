/** Store cross-page inference preferences and context capacity, and centralize language and theme persistence. */
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useLayoutEffect,
  useMemo,
  useState,
  type Dispatch,
  type ReactNode,
  type SetStateAction,
} from 'react';
import {
  loadSettings,
  SETTINGS_KEY,
  type GenerationSettings,
} from './configuration';
import { applyLanguage } from '../../i18n';

interface SettingsContextValue {
  settings: GenerationSettings;
  /** Merge applied preferences and immediately synchronize the theme and local storage. */
  updateSettings: (patch: Partial<GenerationSettings>) => void;
  /** Replace applied preferences with a complete settings object. */
  replaceSettings: Dispatch<SetStateAction<GenerationSettings>>;
  contextSize: number;
  /** Update the context capacity shared by model loading and reloads. */
  setContextSize: Dispatch<SetStateAction<number>>;
}

const SettingsContext = createContext<SettingsContextValue | null>(null);

/** Provide shared preferences through the app shell; pages own their drafts and business requests. */
export function SettingsProvider({ children }: { children: ReactNode }) {
  const [settings, replaceSettings] = useState(loadSettings);
  const [contextSize, setContextSize] = useState(32768);
  const updateSettings = useCallback(
    (patch: Partial<GenerationSettings>) =>
      replaceSettings((current) => ({ ...current, ...patch })),
    [],
  );

  useLayoutEffect(() => {
    const synchronize = () => applyLanguage(settings.language);
    synchronize();
    window.addEventListener('languagechange', synchronize);
    return () => window.removeEventListener('languagechange', synchronize);
  }, [settings.language]);

  useEffect(() => {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
    document.documentElement.dataset.theme = settings.theme;
  }, [settings]);

  const value = useMemo(
    () => ({
      settings,
      replaceSettings,
      updateSettings,
      contextSize,
      setContextSize,
    }),
    [settings, updateSettings, contextSize],
  );
  return (
    <SettingsContext.Provider value={value}>
      {children}
    </SettingsContext.Provider>
  );
}

/** Read shared application settings; must be called within SettingsProvider. */
export function useSettings() {
  const context = useContext(SettingsContext);
  if (!context) throw new Error('SettingsProvider is required');
  return context;
}
