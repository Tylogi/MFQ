/** Rich-text entry point for chat messages, centralizing lazy Markdown loading and streaming-render options. */
import { lazy, Suspense } from 'react';
const Markdown = lazy(() => import('./markdown/Markdown').then((module) => ({ default: module.Markdown })));
/** Render rich-text messages while keeping the plain text readable as the renderer loads. */
export function renderMarkdown(text: string, live = false, normalizeEscapedLineBreaks = false) {
  return <Suspense fallback={<p className='rich-text'>{text}</p>}>
    <Markdown live={live} normalizeEscapedLineBreaks={normalizeEscapedLineBreaks} text={text} />
  </Suspense>;
}
