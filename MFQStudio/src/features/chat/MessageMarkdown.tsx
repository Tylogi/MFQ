/** 聊天富文本入口：统一按需加载 Markdown 与流式渲染参数。 */
import { lazy, Suspense } from 'react';
const Markdown = lazy(() => import('./markdown/Markdown').then((module) => ({ default: module.Markdown })));

/** 渲染消息富文本，加载渲染器期间保留可读正文。 */
export function renderMarkdown(text: string, live = false, normalizeEscapedLineBreaks = false) {
  return <Suspense fallback={<p className='rich-text'>{text}</p>}>
    <Markdown live={live} normalizeEscapedLineBreaks={normalizeEscapedLineBreaks} text={text} />
  </Suspense>;
}
