/** 验证模型文本换行修复不会破坏代码、JSON 和数学表达式。 */
import { describe, expect, it } from 'vitest';
import { normalizeEscapedMarkdownLineBreaks } from './markdownText';

describe('转义 Markdown 换行', () => {
  it('恢复完整转义的段落和列表结构', () => {
    expect(normalizeEscapedMarkdownLineBreaks(String.raw`标题\n\n- 第一项\n- 第二项`)).toBe(
      '标题\n\n- 第一项\n- 第二项',
    );
  });

  it.each([
    JSON.stringify({ text: '第一段\n\n第二段' }),
    '```js\nconst value = "\\n";\n```',
    '真实换行\n保留字面量\\n- 列表',
    String.raw`普通文本\n结束`,
    String.raw`\newcommand{\name}{value}`,
  ])('保留非修复目标 %s', (text) => {
    expect(normalizeEscapedMarkdownLineBreaks(text)).toBe(text);
  });
});
