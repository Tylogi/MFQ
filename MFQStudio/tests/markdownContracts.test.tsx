/** 补充富文本行为与静态样式契约，承接旧 Python Markdown 检查。 */
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { render, screen } from '@testing-library/react';
import { expect, it } from 'vitest';
import { Markdown } from '../src/features/chat/markdown/Markdown';

it('GFM 表格和删除线正常显示，代码中的公式保持原文', () => {
  const view = render(<Markdown text={'| name | value |\n| --- | --- |\n| item | 42 |\n\n~~old~~\n\n`$x$`\n\n```text\n$$y$$\n```'} />);
  expect(screen.getByRole('table')).toHaveTextContent('item');
  expect(view.container.querySelector('del')).toHaveTextContent('old');
  expect(view.container.querySelector('code')).toHaveTextContent('$x$');
  expect(view.container.querySelector('code .katex')).toBeNull();
  expect(view.container.querySelector('pre .katex')).toBeNull();
  expect(view.container.querySelector('pre')).toHaveTextContent('$$y$$');
});

it('保留富文本依赖及公式样式入口', () => {
  const pkg = JSON.parse(readFileSync(resolve('package.json'), 'utf8'));
  for (const dependency of ['dompurify', 'katex', 'marked']) {
    expect(pkg.dependencies[dependency]).toBeTruthy();
  }
  expect(readFileSync(resolve('src/features/chat/markdown/Markdown.tsx'), 'utf8'))
    .toContain("import 'katex/dist/katex.min.css'");
});

it('表格与监控指标保留独立的中性样式约束', () => {
  const styles = [
    'src/features/chat/chat.css', 'src/features/chat/markdown/markdown.css', 'src/shared/styles/overrides.css',
    'src/shared/styles/components.css', 'src/features/runtime/runtime.css',
  ].map((file) => readFileSync(resolve(file), 'utf8')).join('\n');
  expect(styles).toContain('.rich-text table');
  expect(styles).toContain('.runtime-chart polyline');
  expect(styles).not.toContain('.metric-tile::before');
  expect(readFileSync(resolve('src/app/display.tsx'), 'utf8')).not.toContain('metric-tile accent-');
});
