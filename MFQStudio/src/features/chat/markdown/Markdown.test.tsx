/**
 * 验证流式 Markdown 限频、Prism 语法高亮与类名生成、代码块复制交互和不可信 HTML 净化。
 */
import { act, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { Markdown } from './Markdown';

afterEach(() => vi.useRealTimers());

describe('Markdown', () => {
  it('持续输入时按时间窗解析最新文本，结束时无需等待窗口', async () => {
    vi.useFakeTimers();
    const view = render(<Markdown text="first" live />);
    view.rerender(<Markdown text="second" live />);
    await act(() => vi.advanceTimersByTimeAsync(40));
    view.rerender(<Markdown text="third" live />);
    expect(screen.getByText('first')).toBeInTheDocument();
    await act(() => vi.advanceTimersByTimeAsync(40));
    expect(screen.getByText('third')).toBeInTheDocument();
    view.rerender(<Markdown text="final" />);
    expect(screen.getByText('final')).toBeInTheDocument();
    expect(vi.getTimerCount()).toBe(0);
  });

  it('完整和流式输出都净化事件属性、脚本与危险链接', async () => {
    const text =
      '<img src="x" onerror="alert(1)"><script>alert(1)</script>[link](javascript:alert(1))';
    const view = render(<Markdown text={text} live />);
    expect(view.container.querySelector('script')).toBeNull();
    expect(view.container.querySelector('[onerror]')).toBeNull();
    expect(view.container.querySelector('a[href^="javascript:"]')).toBeNull();
    view.rerender(<Markdown text={text} />);
    expect(view.container.querySelector('[onerror]')).toBeNull();
  });

  it('未闭合代码块完成后有一个复制按钮，卸载时清理限频任务', async () => {
    vi.useFakeTimers();
    const view = render(<Markdown text={'```js\nconst a = 1;'} live />);
    expect(screen.queryByRole('button')).not.toBeInTheDocument();
    view.rerender(<Markdown text={'```js\nconst a = 1;\n```'} />);
    expect(screen.getAllByRole('button', { name: 'Copy' })).toHaveLength(1);
    view.rerender(<Markdown text="next" live />);
    view.unmount();
    expect(vi.getTimerCount()).toBe(0);
  });

  it('为常见语言生成标准代码块结构、语言标识与 Prism 语法高亮类名', () => {
    const sample = [
      '```typescript',
      'const count: number = 42;',
      '```',
      '',
      '```python',
      'def greet(name: str):',
      '    return f"Hello, {name}"',
      '```',
      '',
      '```bash',
      'echo "status ok" | grep ok',
      '```',
      '',
      '```json',
      '{"status": "ok", "code": 200}',
      '```',
      '',
      '```rust',
      'fn main() {',
      '    let count: i32 = 42;',
      '}',
      '```',
      '',
      '```cpp',
      'int main() {',
      '    return 0;',
      '}',
      '```',
    ].join('\n');

    const view = render(<Markdown text={sample} />);

    // 检查 TypeScript 代码块结构与高亮
    const tsPre = view.container.querySelector('pre[data-language="typescript"]');
    expect(tsPre).not.toBeNull();
    expect(tsPre?.className).toContain('code-block');
    expect(tsPre?.className).toContain('language-typescript');
    expect(tsPre?.querySelector('.code-lang')?.textContent).toBe('typescript');
    expect(tsPre?.querySelector('.token.keyword')?.textContent).toBe('const');
    expect(tsPre?.querySelector('.token.builtin')?.textContent).toBe('number');
    expect(tsPre?.querySelector('.token.number')?.textContent).toBe('42');

    // 检查 Python 代码块结构与高亮
    const pyPre = view.container.querySelector('pre[data-language="python"]');
    expect(pyPre).not.toBeNull();
    expect(pyPre?.querySelector('.code-lang')?.textContent).toBe('python');
    expect(pyPre?.querySelector('.token.keyword')?.textContent).toBe('def');
    expect(pyPre?.querySelector('.token.function')?.textContent).toBe('greet');

    // 检查 Bash 代码块
    const bashPre = view.container.querySelector('pre[data-language="bash"]');
    expect(bashPre).not.toBeNull();
    expect(bashPre?.querySelector('.code-lang')?.textContent).toBe('bash');
    expect(bashPre?.querySelector('.token.string')?.textContent).toBe('"status ok"');

    // 检查 JSON 代码块结构与高亮
    const jsonPre = view.container.querySelector('pre[data-language="json"]');
    expect(jsonPre).not.toBeNull();
    expect(jsonPre?.querySelector('.code-lang')?.textContent).toBe('json');
    expect(jsonPre?.querySelector('.token.property')?.textContent).toBe('"status"');
    expect(jsonPre?.querySelector('.token.number')?.textContent).toBe('200');

    // 检查 Rust 代码块
    const rustPre = view.container.querySelector('pre[data-language="rust"]');
    expect(rustPre).not.toBeNull();
    expect(rustPre?.querySelector('.token.keyword')?.textContent).toBe('fn');

    // 检查 C++ 代码块
    const cppPre = view.container.querySelector('pre[data-language="cpp"]');
    expect(cppPre).not.toBeNull();
    expect(cppPre?.querySelector('.token.keyword')?.textContent).toBe('int');
  });

  it('未提供语言或未知语言时优雅降级并安全转义 HTML', () => {
    const rawMarkdown = [
      '```',
      'const plain = "text";',
      '<script>alert(1)</script>',
      '```',
      '',
      '```unknownlang',
      'raw <b>bold</b> text',
      '```',
    ].join('\n');

    const view = render(<Markdown text={rawMarkdown} />);

    // 未提供语言时降级为 text 标签并安全转义
    const textPre = view.container.querySelector('pre[data-language="text"]');
    expect(textPre).not.toBeNull();
    expect(textPre?.querySelector('.code-lang')?.textContent).toBe('text');
    expect(textPre?.querySelector('script')).toBeNull();
    expect(textPre?.textContent).toContain('<script>alert(1)</script>');

    // 未知语言时保留原始语言名且安全转义内部 HTML
    const unknownPre = view.container.querySelector('pre[data-language="unknownlang"]');
    expect(unknownPre).not.toBeNull();
    expect(unknownPre?.className).toContain('language-unknownlang');
    expect(unknownPre?.querySelector('.code-lang')?.textContent).toBe('unknownlang');
    expect(unknownPre?.querySelector('b')).toBeNull();
    expect(unknownPre?.textContent).toContain('raw <b>bold</b> text');
  });

  it('点击代码复制按钮调用剪贴板并提供临时状态反馈', async () => {
    vi.useFakeTimers();
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.assign(navigator, {
      clipboard: {
        writeText,
      },
    });

    const view = render(<Markdown text={'```ts\nconst greeting = "hello";\n```'} />);
    const copyButton = screen.getByRole('button', { name: 'Copy' });
    expect(copyButton).toBeInTheDocument();

    fireEvent.click(copyButton);
    await act(async () => {
      await Promise.resolve();
    });

    expect(writeText).toHaveBeenCalledWith('const greeting = "hello";');
    expect(copyButton.textContent).toBe('Copied');
    expect(copyButton.className).toContain('copied');

    await act(() => vi.advanceTimersByTimeAsync(1200));
    expect(copyButton.textContent).toBe('Copy');
    expect(copyButton.className).not.toContain('copied');
  });

  it('完成渲染后 KaTeX 公式渲染正常工作且不破坏 DOM', () => {
    const mathText = [
      '# 公式测试',
      '',
      '行内公式：$E = mc^2$',
      '',
      '块级公式：',
      '$$\\int_0^1 x^2 dx = \\frac{1}{3}$$',
    ].join('\n');

    const view = render(<Markdown text={mathText} />);
    expect(view.container.querySelector('.katex')).not.toBeNull();
    expect(view.container.querySelector('.katex-display')).not.toBeNull();
  });
});
