/** Verify shared settings merging, persistence, and context-capacity behavior across pages. */
import { useTranslation } from 'react-i18next';
import { act, fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { DEFAULT_SETTINGS, SETTINGS_KEY } from './configuration';
import { SettingsProvider, useSettings } from './SettingsProvider';

/** Expose two independent consumers to check shared state and update scope across pages. */
function SettingsConsumer() {
  const { settings, updateSettings, contextSize, setContextSize } =
    useSettings();
  const { t } = useTranslation();
  return (
    <>
      <output aria-label="prompt">{settings.systemPrompt}</output>
      <output aria-label="context">{contextSize}</output>
      <output aria-label="translation">{t('common:settings')}</output>
      <button onClick={() => updateSettings({ theme: 'dark', language: 'en' })}>
        appearance
      </button>
      <button onClick={() => updateSettings({ language: 'zh-CN' })}>
        chinese
      </button>
      <button onClick={() => updateSettings({ language: 'system' })}>
        system
      </button>
      <button onClick={() => setContextSize(8192)}>context</button>
    </>
  );
}

describe('SettingsProvider', () => {
  it('updates mounted consumers, document semantics, and the persisted language after remounting', () => {
    const view = render(
      <SettingsProvider>
        <SettingsConsumer />
      </SettingsProvider>,
    );
    fireEvent.click(screen.getByText('chinese'));
    expect(screen.getByLabelText('translation')).toHaveTextContent('设置');
    expect(document.documentElement.lang).toBe('zh-CN');
    expect(JSON.parse(localStorage.getItem(SETTINGS_KEY)!).language).toBe(
      'zh-CN',
    );
    view.unmount();
    render(
      <SettingsProvider>
        <SettingsConsumer />
      </SettingsProvider>,
    );
    expect(screen.getByLabelText('translation')).toHaveTextContent('设置');
    fireEvent.click(screen.getByText('appearance'));
    expect(screen.getByLabelText('translation')).toHaveTextContent('Settings');
    expect(document.documentElement.lang).toBe('en');
  });

  it('responds to system language changes only while following the system', () => {
    const language = vi
      .spyOn(navigator, 'language', 'get')
      .mockReturnValue('zh-TW');
    render(
      <SettingsProvider>
        <SettingsConsumer />
      </SettingsProvider>,
    );
    expect(screen.getByLabelText('translation')).toHaveTextContent('设置');
    language.mockReturnValue('de-DE');
    act(() => window.dispatchEvent(new Event('languagechange')));
    expect(screen.getByLabelText('translation')).toHaveTextContent('Settings');
    fireEvent.click(screen.getByText('chinese'));
    act(() => window.dispatchEvent(new Event('languagechange')));
    expect(screen.getByLabelText('translation')).toHaveTextContent('设置');
    fireEvent.click(screen.getByText('system'));
    expect(screen.getByLabelText('translation')).toHaveTextContent('Settings');
  });

  it('preserves inference fields when appearance changes and persists the applied settings', () => {
    localStorage.setItem(
      SETTINGS_KEY,
      JSON.stringify({
        ...DEFAULT_SETTINGS,
        systemPrompt: 'retain this prompt',
        language: 'zh',
      }),
    );
    render(
      <SettingsProvider>
        <SettingsConsumer />
      </SettingsProvider>,
    );
    fireEvent.click(screen.getByText('appearance'));
    expect(screen.getByLabelText('prompt')).toHaveTextContent(
      'retain this prompt',
    );
    expect(screen.getByLabelText('translation')).toHaveTextContent('Settings');
    expect(document.documentElement.dataset.theme).toBe('dark');
    expect(
      JSON.parse(localStorage.getItem(SETTINGS_KEY) ?? '{}'),
    ).toMatchObject({
      language: 'en',
      theme: 'dark',
      systemPrompt: 'retain this prompt',
    });
  });

  it('shares the chosen context size across consumers', () => {
    render(
      <SettingsProvider>
        <SettingsConsumer />
        <SettingsConsumer />
      </SettingsProvider>,
    );
    fireEvent.click(screen.getAllByText('context')[0]);
    expect(
      screen.getAllByLabelText('context').map((element) => element.textContent),
    ).toEqual(['8192', '8192']);
  });
});
