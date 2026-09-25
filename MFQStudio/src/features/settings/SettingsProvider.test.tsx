/** 验证共享设置更新的合并、持久化和跨页面上下文容量行为。 */
import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { DEFAULT_SETTINGS, SETTINGS_KEY } from './configuration';
import { SettingsProvider, useSettings } from './SettingsProvider';

/** 暴露两个独立消费者，用于检查跨页面共享状态与修改范围。 */
function SettingsConsumer() {
  const { settings, updateSettings, contextSize, setContextSize, tr } = useSettings();
  return (
    <>
      <output aria-label="prompt">{settings.systemPrompt}</output>
      <output aria-label="context">{contextSize}</output>
      <output aria-label="translation">{tr('设置', 'Settings')}</output>
      <button onClick={() => updateSettings({ theme: 'dark', language: 'en' })}>appearance</button>
      <button onClick={() => setContextSize(8192)}>context</button>
    </>
  );
}

describe('SettingsProvider', () => {
  it('preserves inference fields when appearance changes and persists the applied settings', () => {
    localStorage.setItem(
      SETTINGS_KEY,
      JSON.stringify({ ...DEFAULT_SETTINGS, systemPrompt: 'retain this prompt', language: 'zh' }),
    );
    render(
      <SettingsProvider>
        <SettingsConsumer />
      </SettingsProvider>,
    );
    fireEvent.click(screen.getByText('appearance'));
    expect(screen.getByLabelText('prompt')).toHaveTextContent('retain this prompt');
    expect(screen.getByLabelText('translation')).toHaveTextContent('Settings');
    expect(document.documentElement.dataset.theme).toBe('dark');
    expect(JSON.parse(localStorage.getItem(SETTINGS_KEY) ?? '{}')).toMatchObject({
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
    expect(screen.getAllByLabelText('context').map((element) => element.textContent)).toEqual([
      '8192',
      '8192',
    ]);
  });
});
