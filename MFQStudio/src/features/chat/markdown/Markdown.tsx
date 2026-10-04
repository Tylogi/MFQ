/**
* Markdown rich-text rendering component.
* Parse generated model text with streaming throttling, Prism.js highlighting, code-block copying, and KaTeX math rendering.
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
/** Markdown component props. */
export interface MarkdownProps {
/** Markdown source text to render. */
  text: string;
/** Whether generation is streaming; throttle updates and defer expensive math and DOM interactions while streaming. */
  live?: boolean;
/** Whether fully escaped structural newlines should be restored. */
  normalizeEscapedLineBreaks?: boolean;
}
/** HTML character-escaping map. */
const ESCAPE_HTML_MAP: Record<string, string> = {
  '&': '&amp;',
  '<': '&lt;',
  '>': '&gt;',
  '"': '&quot;',
  "'": '&#39;',
};

/**
* Escape HTML special characters to prevent direct injection of unknown-language or unhighlighted text.
 *
* @param text Plain text to escape
* @returns A safe string with HTML entities escaped
 */
function escapeHtml(text: string): string {
  return text.replace(/[&<>"']/g, (char) => ESCAPE_HTML_MAP[char] || char);
}
/** Common code-language aliases. */
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
* Resolve a Prism-supported grammar and normalize its language name.
 *
* @param lang The original language identifier
* @returns The normalized language name, aliases, and corresponding Prism grammar
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
/** Marked parser configured with Prism syntax highlighting and standard code-block structure. */
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
/** Throttle full Markdown parsing for long answers and show the final text immediately when generation ends. */
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
* Safely render model text as rich content, with streaming throttling, Prism.js highlighting, and KaTeX math enhancement.
 *
* @param props Component properties
* @returns The rendered rich-text DOM structure
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
// Preserve the object reference for identical HTML so throttled updates do not overwrite math and copy-button DOM enhancements.
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
// Recover gracefully without disrupting React DOM lifecycle
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
