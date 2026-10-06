/** Initialize offline translations and resolve the existing persisted language preference. */
import { createInstance } from 'i18next';
import { initReactI18next } from 'react-i18next';
import { loadSettings } from '../features/settings/configuration';
import { namespaces, resources } from './resources';

export type SupportedLanguage = 'en' | 'zh-CN';

/** Normalize saved preferences, including legacy Chinese codes, with English as the fallback. */
export function resolveLanguage(
  preference: string,
  systemLanguage = navigator.language,
): SupportedLanguage {
  const language = preference === 'system' ? systemLanguage : preference;
  return /^zh(?:-|$)/i.test(language) ? 'zh-CN' : 'en';
}

export const i18n = createInstance();
void i18n.use(initReactI18next).init({
  resources,
  lng: resolveLanguage(loadSettings().language),
  supportedLngs: ['en', 'zh-CN'],
  fallbackLng: 'en',
  defaultNS: namespaces,
  ns: namespaces,
  load: 'currentOnly',
  initAsync: false,
  interpolation: { escapeValue: false },
  returnNull: false,
  returnEmptyString: false,
  react: { useSuspense: false },
});

/** Synchronize translations and the document language without changing the saved preference. */
export function applyLanguage(preference: string): void {
  const language = resolveLanguage(preference);
  if (i18n.language !== language) void i18n.changeLanguage(language);
  document.documentElement.lang = language;
}

applyLanguage(loadSettings().language);
