/** Keep application notifications translatable while preserving raw server and user text. */
import type { ParseKeys, TFunction, TOptions } from 'i18next';

export interface LocalizedMessage {
  /** A checked application catalog key, translated only when displayed. */
  key: ParseKeys;
  /** Raw interpolation values, retained across language changes. */
  options?: TOptions;
}

export type DisplayMessage = string | LocalizedMessage;

/** Capture a catalog key and its values for notifications that may outlive the current language. */
export function localized(
  key: ParseKeys,
  options?: TOptions,
): LocalizedMessage {
  return { key, options };
}

/** Translate application messages at render time; never reinterpret raw backend or user text as keys. */
export function displayMessage(message: DisplayMessage, t: TFunction): string {
  return typeof message === 'string'
    ? message
    : t(message.key, message.options);
}
