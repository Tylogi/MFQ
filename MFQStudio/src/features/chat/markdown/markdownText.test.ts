/** Verify that model-text newline normalization preserves code, JSON, and mathematical expressions. */
import { describe, expect, it } from 'vitest';
import { normalizeEscapedMarkdownLineBreaks } from './markdownText';

describe('describes markdownText test behavior 1', () => {
  it('verifies markdownText test behavior 2', () => {
    expect(normalizeEscapedMarkdownLineBreaks(String.raw`Title\n\n- First item\n- Second item`)).toBe(
      'Title\n\n- First item\n- Second item',
    );
  });

  it.each([
    JSON.stringify({ text: 'First paragraph\n\nSecond paragraph' }),
    '```js\nconst value = "\\n";\n```',
    'Real line break\nKeep literal\\n- list',
    String.raw`Plain text\nEnd`,
    String.raw`\newcommand{\name}{value}`,
  ])('verifies parameterized behavior %s', (text) => {
    expect(normalizeEscapedMarkdownLineBreaks(text)).toBe(text);
  });
});
