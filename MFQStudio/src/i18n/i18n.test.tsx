/** Verify catalog completeness, language rules, interpolation, fallback, and live notification translation. */
import { createInstance } from 'i18next';
import { act, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { i18n, applyLanguage, resolveLanguage } from './index';
import { resources } from './resources';
import { localized } from './messages';
import { ToastItem } from '../shared/ui/ToastItem';

/** Flatten catalog entries for structural comparison without assuming identical plural categories. */
function entries(
  value: Record<string, unknown>,
  prefix = '',
): Record<string, string> {
  return Object.fromEntries(
    Object.entries(value).flatMap(([key, child]) => {
      const name = prefix + key;
      return typeof child === 'string'
        ? [[name, child]]
        : Object.entries(entries(child as Record<string, unknown>, name + '.'));
    }),
  );
}

describe('offline internationalization', () => {
  it.each([
    ['zh', 'en-US', 'zh-CN'],
    ['zh-CN', 'en-US', 'zh-CN'],
    ['en', 'zh-CN', 'en'],
    ['system', 'zh-HK', 'zh-CN'],
    ['system', 'fr-FR', 'en'],
    ['unsupported', 'zh-CN', 'en'],
  ])(
    'resolves %s with system locale %s to %s',
    (preference, system, expected) => {
      expect(resolveLanguage(preference, system)).toBe(expected);
    },
  );

  it('keeps catalogs complete with matching interpolation variables and no empty translations', () => {
    const english = entries(resources.en);
    const chinese = entries(resources['zh-CN']);
    const normalize = (key: string) => key.replace(/_one$/, '_other');
    expect([...new Set(Object.keys(english).map(normalize))].sort()).toEqual(
      Object.keys(chinese).sort(),
    );
    const variables = (text: string) =>
      [...text.matchAll(/\{\{\s*([^},]+)(?:,[^}]*)?\}\}/g)]
        .map((match) => match[1].trim())
        .sort();
    for (const [key, text] of Object.entries(english)) {
      const translated = chinese[normalize(key)];
      expect(text.trim(), key).not.toBe('');
      expect(translated.trim(), key).not.toBe('');
      expect(variables(translated), key).toEqual(variables(text));
    }
  });

  it('handles English plurals and Chinese counts using raw quantities', () => {
    const en = i18n.getFixedT('en');
    const zh = i18n.getFixedT('zh-CN');
    expect(en('runtime:memoryHierarchy.models', { count: 0 })).toBe('0 models');
    expect(en('runtime:memoryHierarchy.models', { count: 1 })).toBe('1 model');
    expect(en('runtime:memoryHierarchy.models', { count: 2 })).toBe('2 models');
    expect(zh('runtime:memoryHierarchy.models', { count: 2 })).toBe('2 个模型');
    expect(
      en('models:repositoryFiles.downloadSelectedFiles', { count: 1 }),
    ).toBe('Download 1 selected file');
  });

  it('uses bundled English fallback when a translated entry is missing', async () => {
    const instance = createInstance();
    await instance.init({
      initAsync: false,
      lng: 'zh-CN',
      fallbackLng: 'en',
      defaultNS: 'common',
      resources: { en: { common: { save: 'Save' } }, 'zh-CN': { common: {} } },
    });
    expect(instance.t('common:save')).toBe('Save');
  });

  it('retranslates a visible notification without changing raw content or dismissing it', () => {
    const onDismiss = vi.fn();
    render(
      <ToastItem
        onDismiss={onDismiss}
        toast={{
          id: 'saved',
          revision: 1,
          type: 'success',
          duration: 0,
          title: 'Qwen <custom>',
          message: localized(
            'settings:settingsRoute.settingsAppliedSuccessfully',
          ),
        }}
      />,
    );
    expect(screen.getByRole('status')).toHaveTextContent(
      'Settings applied successfully',
    );
    act(() => applyLanguage('zh-CN'));
    expect(screen.getByRole('status')).toHaveTextContent('设置已应用');
    expect(screen.getByRole('status')).toHaveTextContent('Qwen <custom>');
    expect(screen.getByRole('button', { name: '关闭通知' })).toBeVisible();
    expect(onDismiss).not.toHaveBeenCalled();
  });

  it('renders interpolated user text as text rather than HTML', () => {
    const name = '<img src=x onerror=alert(1)>';
    const message = localized('chat:useChatAttachments.unsupportedAttachment', {
      name,
    });
    const view = render(
      <ToastItem
        onDismiss={vi.fn()}
        toast={{
          id: 'attachment',
          revision: 1,
          type: 'error',
          duration: 0,
          message,
        }}
      />,
    );
    expect(screen.getByRole('alert')).toHaveTextContent(name);
    expect(view.container.querySelector('img')).toBeNull();
  });
});
