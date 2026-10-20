import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { expect, it } from 'vitest';

it('keeps chat mode and reasoning selects the same height as toolbar buttons', () => {
  const style = document.createElement('style');
  style.textContent = ['src/features/chat/chat.css', 'src/shared/styles/selects.css']
    .map((path) => readFileSync(resolve(path), 'utf8'))
    .join('\n');
  const toolbar = document.createElement('div');
  toolbar.className = 'composer-toolbar';
  const button = document.createElement('button');
  toolbar.append(button);
  for (const text of ['文本', '半双工', '全双工', 'Full duplex', 'Standard']) {
    const select = document.createElement('select');
    select.add(new Option(text, text));
    toolbar.append(select);
  }
  document.head.append(style);
  document.body.append(toolbar);
  try {
    const buttonHeight = getComputedStyle(button).height;
    expect(buttonHeight).toBe('31px');
    for (const select of toolbar.querySelectorAll('select')) {
      const computed = getComputedStyle(select);
      expect(computed.height).toBe(buttonHeight);
      expect(computed.minHeight).toBe(buttonHeight);
      expect(computed.paddingTop).toBe('0px');
      expect(computed.paddingBottom).toBe('0px');
      expect(computed.paddingRight).toBe('25px');
      expect(computed.borderRadius).toBe('8px');
      expect(computed.fontSize).toBe('11px');
    }
  } finally {
    toolbar.remove();
    style.remove();
  }
});
