/**
* Verify streaming Markdown throttling, Prism highlighting and class names, code-block copy interactions, and sanitization of untrusted HTML.
 */
import { act, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { Markdown } from './Markdown';

afterEach(() => vi.useRealTimers());

describe('Markdown', () => {
  it('verifies Markdown test behavior 1', async () => {
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

  it('verifies Markdown test behavior 2', async () => {
    const text =
      '<img src="x" onerror="alert(1)"><script>alert(1)</script>[link](javascript:alert(1))';
    const view = render(<Markdown text={text} live />);
    expect(view.container.querySelector('script')).toBeNull();
    expect(view.container.querySelector('[onerror]')).toBeNull();
    expect(view.container.querySelector('a[href^="javascript:"]')).toBeNull();
    view.rerender(<Markdown text={text} />);
    expect(view.container.querySelector('[onerror]')).toBeNull();
  });

  it('verifies Markdown test behavior 3', async () => {
    vi.useFakeTimers();
    const view = render(<Markdown text={'```js\nconst a = 1;'} live />);
    expect(screen.queryByRole('button')).not.toBeInTheDocument();
    view.rerender(<Markdown text={'```js\nconst a = 1;\n```'} />);
    expect(screen.getAllByRole('button', { name: 'Copy' })).toHaveLength(1);
    view.rerender(<Markdown text="next" live />);
    view.unmount();
    expect(vi.getTimerCount()).toBe(0);
  });

  it('verifies Markdown test behavior 4', () => {
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
// Check TypeScript code-block structure and highlighting
    const tsPre = view.container.querySelector('pre[data-language="typescript"]');
    expect(tsPre).not.toBeNull();
    expect(tsPre?.className).toContain('code-block');
    expect(tsPre?.className).toContain('language-typescript');
    expect(tsPre?.querySelector('.code-lang')?.textContent).toBe('typescript');
    expect(tsPre?.querySelector('.token.keyword')?.textContent).toBe('const');
    expect(tsPre?.querySelector('.token.builtin')?.textContent).toBe('number');
    expect(tsPre?.querySelector('.token.number')?.textContent).toBe('42');
// Check Python code-block structure and highlighting
    const pyPre = view.container.querySelector('pre[data-language="python"]');
    expect(pyPre).not.toBeNull();
    expect(pyPre?.querySelector('.code-lang')?.textContent).toBe('python');
    expect(pyPre?.querySelector('.token.keyword')?.textContent).toBe('def');
    expect(pyPre?.querySelector('.token.function')?.textContent).toBe('greet');
// Check Bash code block
    const bashPre = view.container.querySelector('pre[data-language="bash"]');
    expect(bashPre).not.toBeNull();
    expect(bashPre?.querySelector('.code-lang')?.textContent).toBe('bash');
    expect(bashPre?.querySelector('.token.string')?.textContent).toBe('"status ok"');
// Check JSON code-block structure and highlighting
    const jsonPre = view.container.querySelector('pre[data-language="json"]');
    expect(jsonPre).not.toBeNull();
    expect(jsonPre?.querySelector('.code-lang')?.textContent).toBe('json');
    expect(jsonPre?.querySelector('.token.property')?.textContent).toBe('"status"');
    expect(jsonPre?.querySelector('.token.number')?.textContent).toBe('200');
// Check Rust code block
    const rustPre = view.container.querySelector('pre[data-language="rust"]');
    expect(rustPre).not.toBeNull();
    expect(rustPre?.querySelector('.token.keyword')?.textContent).toBe('fn');
// Check C++ code block
    const cppPre = view.container.querySelector('pre[data-language="cpp"]');
    expect(cppPre).not.toBeNull();
    expect(cppPre?.querySelector('.token.keyword')?.textContent).toBe('int');
  });

  it('verifies Markdown test behavior 5', () => {
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
// Fall back to a text label and escape safely when no language is provided
    const textPre = view.container.querySelector('pre[data-language="text"]');
    expect(textPre).not.toBeNull();
    expect(textPre?.querySelector('.code-lang')?.textContent).toBe('text');
    expect(textPre?.querySelector('script')).toBeNull();
    expect(textPre?.textContent).toContain('<script>alert(1)</script>');
// Preserve the original language name and safely escape embedded HTML for unknown languages
    const unknownPre = view.container.querySelector('pre[data-language="unknownlang"]');
    expect(unknownPre).not.toBeNull();
    expect(unknownPre?.className).toContain('language-unknownlang');
    expect(unknownPre?.querySelector('.code-lang')?.textContent).toBe('unknownlang');
    expect(unknownPre?.querySelector('b')).toBeNull();
    expect(unknownPre?.textContent).toContain('raw <b>bold</b> text');
  });

  it('verifies Markdown test behavior 6', async () => {
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

  it('verifies Markdown test behavior 7', () => {
    const mathText = [
      '# Formula test',
      '',
      'Inline formula：$E = mc^2$',
      '',
      'Block formula：',
      '$$\\int_0^1 x^2 dx = \\frac{1}{3}$$',
    ].join('\n');

    const view = render(<Markdown text={mathText} />);
    expect(view.container.querySelector('.katex')).not.toBeNull();
    expect(view.container.querySelector('.katex-display')).not.toBeNull();
  });
});
