/** 验证滚动跟随、生命周期清理及推理展示，CSS 仅保留布局边界检查。 */
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { act, fireEvent, render, screen } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
import { useChatAutoScroll } from '../src/features/chat/hooks/useChatAutoScroll';
import { SavedMessageList } from '../src/features/chat/SavedMessageList';
import { StreamingMessage } from '../src/features/chat/components/StreamingMessage';
import { GenerationController } from '../src/features/chat/state/generationController';
import { SettingsPage } from '../src/features/settings/SettingsPage';
import { DEFAULT_SETTINGS } from '../src/features/settings/configuration';
import { TooltipProvider } from '../src/shared/ui/Tooltip';
import type { Message } from '../src/shared/api/types';

const observers: { resize: () => void; observe: ReturnType<typeof vi.fn>; disconnect: ReturnType<typeof vi.fn> }[] = [];
const frames = new Map<number, FrameRequestCallback>();
let nextFrame = 0;
let scroll: ReturnType<typeof useChatAutoScroll>;

/** 提供真实 DOM ref 和滚动事件，由测试控制浏览器布局尺寸。 */
function ScrollSurface({ sessionId = 'a', enabled = true }) {
  scroll = useChatAutoScroll(sessionId, enabled);
  return <div data-testid="scroller" ref={scroll.scrollerRef} onScroll={scroll.handleScroll}><div>messages</div></div>;
}

/** 执行单帧回调，验证合帧与用户操作之间的竞态。 */
function flushFrames() {
  act(() => {
    const pending = [...frames.values()];
    frames.clear();
    pending.forEach((callback) => callback(0));
  });
}

beforeEach(() => {
  frames.clear();
  observers.length = 0;
  nextFrame = 0;
  vi.stubGlobal('requestAnimationFrame', vi.fn((callback: FrameRequestCallback) => { frames.set(++nextFrame, callback); return nextFrame; }));
  vi.stubGlobal('cancelAnimationFrame', vi.fn((id: number) => frames.delete(id)));
  vi.stubGlobal('ResizeObserver', class {
    observe = vi.fn();
    disconnect = vi.fn();
    constructor(resize: () => void) { observers.push({ resize, observe: this.observe, disconnect: this.disconnect }); }
  });
});

it('仅距尾部 80px 内跟随，内容增长合帧，向上阅读时不抢滚动位置', () => {
  render(<ScrollSurface />);
  const scroller = screen.getByTestId('scroller');
  Object.defineProperties(scroller, { scrollHeight: { configurable: true, value: 1000 }, clientHeight: { value: 200 } });
  expect(observers[0].observe.mock.calls).toEqual([[scroller], [scroller.firstElementChild]]);
  flushFrames();
  expect(scroller.scrollTop).toBe(1000);
  scroller.scrollTop = 719;
  fireEvent.scroll(scroller);
  act(() => observers[0].resize());
  expect(scroll.following).toBe(false);
  expect(frames.size).toBe(0);
  expect(scroller.scrollTop).toBe(719);
  scroller.scrollTop = 720;
  fireEvent.scroll(scroller);
  expect(scroll.following).toBe(true);
  act(() => { observers[0].resize(); observers[0].resize(); });
  expect(frames.size).toBe(1);
  flushFrames();
  expect(scroller.scrollTop).toBe(1000);
});

it('待执行帧尊重用户暂停，回到底部恢复后续跟随', () => {
  render(<ScrollSurface />);
  const scroller = screen.getByTestId('scroller');
  Object.defineProperties(scroller, { scrollHeight: { configurable: true, value: 1000 }, clientHeight: { value: 200 } });
  scroller.scrollTop = 100;
  fireEvent.scroll(scroller);
  flushFrames();
  expect(scroller.scrollTop).toBe(100);
  act(() => scroll.scrollToBottom());
  expect(scroll.following).toBe(true);
  expect(scroller.scrollTop).toBe(1000);
  Object.defineProperty(scroller, 'scrollHeight', { value: 1200 });
  act(() => observers[0].resize());
  flushFrames();
  expect(scroller.scrollTop).toBe(1200);
});

it('切会话、禁用和卸载断开观察并取消待执行帧，重新启用恢复跟随', () => {
  const view = render(<ScrollSurface enabled={false} />);
  expect(observers).toHaveLength(0);
  view.rerender(<ScrollSurface />);
  expect(frames.size).toBe(1);
  view.rerender(<ScrollSurface sessionId="b" />);
  expect(observers[0].disconnect).toHaveBeenCalledOnce();
  expect(cancelAnimationFrame).toHaveBeenCalledWith(1);
  expect(frames.size).toBe(1);
  view.rerender(<ScrollSurface sessionId="b" enabled={false} />);
  expect(observers[1].disconnect).toHaveBeenCalledOnce();
  expect(frames.size).toBe(0);
  view.rerender(<ScrollSurface sessionId="b" />);
  expect(scroll.following).toBe(true);
  view.unmount();
  expect(observers[2].disconnect).toHaveBeenCalledOnce();
  expect(frames.size).toBe(0);
});

it('历史消息编辑展示草稿与保存入口，助手消息提供重新生成', () => {
  const user = { id: 'user', role: 'user', parts: [{ type: 'text', text: 'original' }], created_at: '' } as Message;
  const assistant = { ...user, id: 'assistant', role: 'assistant' } as Message;
  const setEditDraft = vi.fn();
  const actions = { saveEdit: vi.fn(), copyMessage: vi.fn(), regenerate: vi.fn(), executeToolCalls: vi.fn() };
  const props = { messages: [user, assistant], responses: {}, mcpTools: [], busy: false, tr: (_zh: string, en: string) => en, setEditDraft, actions };
  const view = render(<TooltipProvider><SavedMessageList {...props} editDraft={null} /></TooltipProvider>);
  expect(screen.getAllByRole('button', { name: 'Edit' })).toHaveLength(1);
  expect(screen.getAllByRole('button', { name: 'Regenerate' })).toHaveLength(1);
  fireEvent.click(screen.getByRole('button', { name: 'Edit' }));
  expect(setEditDraft).toHaveBeenCalledWith({ messageId: user.id, text: 'original' });
  fireEvent.click(screen.getByRole('button', { name: 'Regenerate' }));
  expect(actions.regenerate).toHaveBeenCalledWith(assistant);
  view.rerender(<TooltipProvider><SavedMessageList {...props} editDraft={{ messageId: user.id, text: 'revised' }} /></TooltipProvider>);
  expect(screen.getByRole('textbox', { name: 'Message' })).toHaveValue('revised');
  expect(screen.queryByRole('button', { name: 'Save as branch' })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Save' }));
  expect(actions.saveEdit).toHaveBeenCalledWith(user);
});

it('历史推理默认折叠，流式推理默认展开且只显示当前会话', () => {
  const controller = new GenerationController({ onSynchronized: vi.fn(), onSessionState: vi.fn() });
  vi.spyOn(controller, 'getSnapshot').mockReturnValue({ phase: 'streaming', sessionId: 'a', live: { reasoning: 'live reasoning', text: '', tools: [] }, error: null, recoveryNeeded: false });
  const message = { id: 'answer', role: 'assistant', parts: [{ type: 'reasoning', text: 'saved reasoning' }], created_at: '' } as Message;
  const tr = (_zh: string, en: string) => en;
  const view = render(<TooltipProvider>
    <SavedMessageList messages={[message]} responses={{}} mcpTools={[]} busy={false} tr={tr} editDraft={null} setEditDraft={vi.fn()} actions={{ saveEdit: vi.fn(), copyMessage: vi.fn(), regenerate: vi.fn(), executeToolCalls: vi.fn() }} />
    <StreamingMessage controller={controller} sessionId="a" tr={tr} />
  </TooltipProvider>);
  expect(screen.getByText('Reasoning').closest('details')).not.toHaveAttribute('open');
  expect(screen.getByText('Thinking').closest('details')).toHaveAttribute('open');
  view.rerender(<StreamingMessage controller={controller} sessionId="b" tr={tr} />);
  expect(screen.queryByText('Thinking')).not.toBeInTheDocument();
});

it('默认生成限制为 4096，设置界面允许最高 65536', () => {
  render(<SettingsPage tr={(_zh, en) => en} settingsDraft={{ ...DEFAULT_SETTINGS, inheritModelDefaults: false }} setSettingsDraft={vi.fn()} mtpAvailable={false} presetManager={null} contextSize={4096} setContextSize={vi.fn()} busy={false} hasStudio={false} actions={{ setModelDefaultInheritance: vi.fn(), applyPreset: vi.fn(), reloadRuntime: vi.fn(), exportStudioData: vi.fn(), importStudioData: vi.fn(), openServerPage: vi.fn(), resetSettingsDraft: vi.fn(), saveSettings: vi.fn() }} />);
  const limit = screen.getAllByRole('spinbutton').find((input) => input.getAttribute('max') === '65536');
  expect(limit).toHaveValue(4096);
  expect(limit).toHaveAttribute('min', '1');
});

it('聊天容器拥有垂直滚动，推理正文不引入独立限高滚动区', () => {
  const style = document.createElement('style');
  style.textContent = readFileSync(resolve('src/features/chat/chat.css'), 'utf8');
  document.head.append(style);
  try {
    const rules = Array.from(style.sheet!.cssRules).filter((rule): rule is CSSStyleRule => rule instanceof CSSStyleRule);
    const declarations = (selector: string) => rules.filter((rule) => rule.selectorText === selector).map((rule) => rule.style);
    expect(declarations('.chat-view').length).toBeGreaterThan(0);
    expect(declarations('.message-list').length).toBeGreaterThan(0);
    expect(declarations('.message-scroller').some((rule) => rule.getPropertyValue('overflow-y') === 'auto')).toBe(true);
    expect(declarations('.reasoning .rich-text').length).toBeGreaterThan(0);
    for (const selector of ['.reasoning', '.reasoning .rich-text']) {
      expect(declarations(selector).length).toBeGreaterThan(0);
      for (const rule of declarations(selector)) {
        expect(rule.getPropertyValue('max-height')).toBe('');
        expect(rule.getPropertyValue('overflow-y')).toBe('');
        expect(rule.getPropertyValue('overflow')).toBe('');
      }
    }
  } finally { style.remove(); }
});
