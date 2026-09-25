/**
 * Markdown 富文本渲染组件。
 * 负责模型生成文本的流式限频解析、Prism.js 语法高亮、代码块复制与 KaTeX 数学公式渲染。
 */
import DOMPurify from 'dompurify';
import renderMathInElement from 'katex/contrib/auto-render';
import { Marked } from 'marked';
import Prism from 'prismjs';
import 'prismjs/components/prism-bash.js';
import 'prismjs/components/prism-c.js';
import 'prismjs/components/prism-cpp.js';
import 'prismjs/components/prism-go.js';
import 'prismjs/components/prism-java.js';
import 'prismjs/components/prism-json.js';
import 'prismjs/components/prism-markdown.js';
import 'prismjs/components/prism-python.js';
import 'prismjs/components/prism-rust.js';
import 'prismjs/components/prism-sql.js';
import 'prismjs/components/prism-typescript.js';
import 'prismjs/components/prism-yaml.js';
import { memo, useEffect, useMemo, useRef, useState } from 'react';
import { normalizeEscapedMarkdownLineBreaks } from './markdownText';
import 'katex/dist/katex.min.css';
import './markdown.css';

/** Markdown 组件入参属性定义。 */
export interface MarkdownProps {
  /** 待渲染的 Markdown 源码文本。 */
  text: string;
  /** 是否处于流式生成状态，流式阶段启用限频且推迟高开销公式与 DOM 交互。 */
  live?: boolean;
  /** 是否需要还原完全转义的结构性换行。 */
  normalizeEscapedLineBreaks?: boolean;
}

/** HTML 字符转义映射表。 */
const ESCAPE_HTML_MAP: Record<string, string> = {
  '&': '&amp;',
  '<': '&lt;',
  '>': '&gt;',
  '"': '&quot;',
  "'": '&#39;',
};

/**
 * 安全转义 HTML 特殊字符，防止未知语言或未高亮文本被直接注入解析。
 *
 * @param text 需要转义的纯文本
 * @returns 经过 HTML 实体转义的安全字符串
 */
function escapeHtml(text: string): string {
  return text.replace(/[&<>"']/g, (char) => ESCAPE_HTML_MAP[char] || char);
}

/** 常用代码语言别名映射。 */
const LANGUAGE_ALIASES: Record<string, string> = {
  js: 'javascript',
  ts: 'typescript',
  py: 'python',
  sh: 'bash',
  shell: 'bash',
  zsh: 'bash',
  'c++': 'cpp',
  yml: 'yaml',
  rs: 'rust',
  md: 'markdown',
  html: 'markup',
  xml: 'markup',
  svg: 'markup',
};

/**
 * 解析并匹配 Prism 支持的语法定义与规范化语言名称。
 *
 * @param lang 原始语言标记字符串
 * @returns 规范化的语言名、别名和对应的 Prism 语法定义
 */
function resolveLanguage(lang?: string): {
  rawLang: string;
  normalizedLang: string;
  grammar: Prism.Grammar | null;
} {
  const rawLang = (lang || '').trim().toLowerCase().split(/\s+/)[0] || '';
  if (!rawLang) {
    return { rawLang: '', normalizedLang: 'text', grammar: null };
  }
  const alias = LANGUAGE_ALIASES[rawLang] || rawLang;
  const grammar = Prism.languages[alias] || Prism.languages[rawLang] || null;
  return {
    rawLang,
    normalizedLang: alias,
    grammar,
  };
}

/** 配置了 Prism 语法高亮和标准代码块结构的 Marked 解析器实例。 */
const customMarked = new Marked({
  breaks: true,
  gfm: true,
  renderer: {
    code({ text, lang }: { text: string; lang?: string }) {
      const { rawLang, normalizedLang, grammar } = resolveLanguage(lang);
      const displayLang = rawLang || 'text';
      const langClass = rawLang ? `language-${rawLang}` : 'language-text';
      const canonicalClass = normalizedLang && normalizedLang !== rawLang ? ` language-${normalizedLang}` : '';
      const classes = `code-block ${langClass}${canonicalClass}`.trim();
      const highlighted = grammar ? Prism.highlight(text, grammar, normalizedLang) : escapeHtml(text);

      return (
        `<pre class="${classes}" data-language="${displayLang}">` +
        `<div class="code-header"><span class="code-lang">${escapeHtml(displayLang)}</span></div>` +
        `<code class="${langClass}">${highlighted}</code>` +
        `</pre>`
      );
    },
  },
});

/** 限制长回答的全量 Markdown 解析频率，并在结束生成时立即展示最终文本。 */
function useStreamingText(text: string, live: boolean): string {
  const [displayed, setDisplayed] = useState(text);
  const latest = useRef(text);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  latest.current = text;
  useEffect(() => {
    if (!live) {
      if (timer.current !== null) clearTimeout(timer.current);
      timer.current = null;
      setDisplayed(text);
    } else if (timer.current === null && text !== displayed) {
      timer.current = setTimeout(() => {
        timer.current = null;
        setDisplayed(latest.current);
      }, 80);
    }
  }, [text, live, displayed]);
  useEffect(
    () => () => {
      if (timer.current !== null) clearTimeout(timer.current);
    },
    [],
  );
  return live ? displayed : text;
}

/**
 * 将模型文本安全渲染为富文本；支持流式限频、Prism.js 代码语法高亮与 KaTeX 数学公式增强。
 *
 * @param props 组件入参
 * @returns 渲染后的富文本 DOM 结构
 */
export const Markdown = memo(function Markdown({
  text,
  live = false,
  normalizeEscapedLineBreaks = false,
}: MarkdownProps) {
  const rootRef = useRef<HTMLDivElement>(null);
  const displayedText = useStreamingText(text, live);
  const markdown = useMemo(
    () =>
      normalizeEscapedLineBreaks
        ? normalizeEscapedMarkdownLineBreaks(displayedText)
        : displayedText,
    [normalizeEscapedLineBreaks, displayedText],
  );
  const html = useMemo(
    () => DOMPurify.sanitize(customMarked.parse(markdown) as string),
    [markdown],
  );
  // 保持相同 HTML 的对象引用，避免限频状态更新覆盖公式和复制按钮的 DOM 增强。
  const markup = useMemo(() => ({ __html: html }), [html]);

  useEffect(() => {
    if (live || !rootRef.current) return;
    try {
      renderMathInElement(rootRef.current, {
        delimiters: [
          { left: '$$', right: '$$', display: true },
          { left: '\\[', right: '\\]', display: true },
          { left: '\\(', right: '\\)', display: false },
          { left: '$', right: '$', display: false },
        ],
        ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'],
        ignoredClasses: ['katex', 'katex-display', 'katex-html', 'katex-mathml'],
        strict: 'ignore',
        throwOnError: false,
        trust: false,
      });
    } catch {
      // 容错处理：不破坏 React DOM 挂载生命周期
    }
    const buttons: HTMLButtonElement[] = [];
    const timers = new Set<ReturnType<typeof setTimeout>>();
    let disposed = false;
    for (const block of rootRef.current.querySelectorAll('pre')) {
      const source = block.querySelector('code')?.textContent || block.textContent || '';
      const button = document.createElement('button');
      button.className = 'code-copy';
      button.type = 'button';
      button.textContent = 'Copy';
      button.addEventListener('click', () => {
        void Promise.resolve()
          .then(() => navigator.clipboard.writeText(source))
          .then(() => {
            if (disposed) return;
            button.textContent = 'Copied';
            button.classList.add('copied');
            const timer = setTimeout(() => {
              button.textContent = 'Copy';
              button.classList.remove('copied');
              timers.delete(timer);
            }, 1200);
            timers.add(timer);
          })
          .catch(() => {
            if (!disposed) button.textContent = 'Copy failed';
          });
      });
      const header = block.querySelector('.code-header');
      if (header) {
        header.append(button);
      } else {
        block.append(button);
      }
      buttons.push(button);
    }
    return () => {
      disposed = true;
      timers.forEach(clearTimeout);
      buttons.forEach((button) => button.remove());
    };
  }, [html, live]);

  return <div className="rich-text" dangerouslySetInnerHTML={markup} ref={rootRef} />;
});
