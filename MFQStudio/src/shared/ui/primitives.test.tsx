/** 验证模态弹窗、工具提示和即时设置开关的键盘及读屏行为。 */
import { useRef, useState } from 'react';
import { act, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { Dialog } from './Dialog';
import { Switch } from './Switch';
import { Tooltip, TooltipProvider } from './Tooltip';
import { ToastContainer } from './Toast';
import { toast, useToastStore } from '../../stores/toastStore';

beforeEach(() => {
  // jsdom 无布局观测器；本组仅验证焦点、键盘和无障碍语义。
  vi.stubGlobal(
    'ResizeObserver',
    class {
      observe = vi.fn();
      unobserve = vi.fn();
      disconnect = vi.fn();
    },
  );
});

/** 模拟两个业务入口共享一个受控弹窗，并支持显式指定焦点返回位置。 */
function DialogFixture({ explicitTarget = false }: { explicitTarget?: boolean }) {
  const [open, setOpen] = useState(false);
  const returnFocusRef = useRef<HTMLButtonElement | null>(null);
  return (
    <>
      <button type="button" ref={returnFocusRef} onClick={() => setOpen(true)}>
        入口一
      </button>
      <button type="button" onClick={() => setOpen(true)}>
        入口二
      </button>
      <Dialog
        open={open}
        onOpenChange={setOpen}
        title="模型目录"
        description="选择服务器上的目录"
        closeLabel="关闭目录"
        returnFocusRef={explicitTarget ? returnFocusRef : undefined}
      >
        <input aria-label="目录路径" />
        <button type="button" onClick={() => setOpen(false)}>
          确认目录
        </button>
      </Dialog>
    </>
  );
}

describe('Dialog', () => {
  it('关联标题和描述，Esc 关闭后返回本次打开的入口', async () => {
    const user = userEvent.setup();
    render(<DialogFixture />);
    for (const name of ['入口一', '入口二']) {
      const trigger = screen.getByRole('button', { name });
      await user.click(trigger);
      const dialog = screen.getByRole('dialog', { name: '模型目录' });
      expect(dialog).toHaveAccessibleDescription('选择服务器上的目录');
      expect(dialog).toContainElement(document.activeElement as HTMLElement);
      await user.keyboard('{Escape}');
      await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
      await waitFor(() => expect(trigger).toHaveFocus());
    }
  });

  it('Tab 和 Shift+Tab 不离开模态内容', async () => {
    const user = userEvent.setup();
    render(<DialogFixture />);
    await user.click(screen.getByRole('button', { name: '入口一' }));
    expect(screen.getByRole('button', { name: '关闭目录' })).toHaveFocus();
    await user.tab({ shift: true });
    expect(screen.getByRole('button', { name: '确认目录' })).toHaveFocus();
    await user.tab();
    expect(screen.getByRole('button', { name: '关闭目录' })).toHaveFocus();
  });

  it('支持业务显式指定异步打开前保存的焦点目标', async () => {
    const user = userEvent.setup();
    render(<DialogFixture explicitTarget />);
    await user.click(screen.getByRole('button', { name: '入口二' }));
    await user.click(screen.getByRole('button', { name: '关闭目录' }));
    await waitFor(() => expect(screen.getByRole('button', { name: '入口一' })).toHaveFocus());
  });
});
describe('Switch', () => {
  it('暴露名称和受控状态，空格键触发更新且不提交表单', async () => {
    const user = userEvent.setup();
    const change = vi.fn();
    const submit = vi.fn((event: React.FormEvent) => event.preventDefault());
    const view = render(
      <form onSubmit={submit}>
        <Switch label="自动跟随" checked={false} onCheckedChange={change} />
      </form>,
    );
    await user.tab();
    await user.keyboard(' ');
    expect(change).toHaveBeenCalledWith(true);
    expect(submit).not.toHaveBeenCalled();
    view.rerender(<Switch label="自动跟随" checked onCheckedChange={change} />);
    expect(screen.getByRole('switch', { name: '自动跟随' })).toBeChecked();
  });

  it('禁用开关不触发设置更新', async () => {
    const user = userEvent.setup();
    const change = vi.fn();
    render(<Switch label="自动跟随" checked={false} onCheckedChange={change} disabled />);
    await user.click(screen.getByRole('switch', { name: '自动跟随' }));
    expect(change).not.toHaveBeenCalled();
  });
});

describe('Tooltip', () => {
  it('键盘聚焦原生按钮时显示提示，Esc 可关闭', async () => {
    const user = userEvent.setup();
    render(
      <TooltipProvider delayDuration={0}>
        <Tooltip content="重新生成回答">
          <button type="button" aria-label="重新生成">
            R
          </button>
        </Tooltip>
      </TooltipProvider>,
    );
    await user.tab();
    expect(await screen.findByRole('tooltip')).toHaveTextContent('重新生成回答');
    expect(screen.getByRole('button', { name: '重新生成' })).toHaveFocus();
    await user.keyboard('{Escape}');
    await waitFor(() => expect(screen.queryByRole('tooltip')).not.toBeInTheDocument());
  });
});

describe('Toast', () => {
  beforeEach(() => {
    useToastStore.getState().clearToasts();
  });

  it('展示错误与成功通知，支持无障碍语义和手动关闭', async () => {
    const user = userEvent.setup();
    render(<ToastContainer />);

    expect(screen.queryByRole('region', { name: '通知提示' })).not.toBeInTheDocument();

    toast.error('模型加载失败', { title: '错误' });
    toast.success('配置已保存');

    expect(await screen.findByRole('alert')).toHaveTextContent('模型加载失败');
    expect(screen.getByText('错误')).toBeInTheDocument();
    expect(screen.getByRole('status')).toHaveTextContent('配置已保存');

    const closeButtons = screen.getAllByRole('button', { name: '关闭通知' });
    expect(closeButtons).toHaveLength(2);

    await user.click(closeButtons[0]);
    await waitFor(() => expect(screen.queryByRole('alert')).not.toBeInTheDocument());
    expect(screen.getByRole('status')).toHaveTextContent('配置已保存');
  });

  it('到达 duration 时自动销毁通知', async () => {
    vi.useFakeTimers();
    render(<ToastContainer />);

    act(() => {
      toast.info('临时提示', { duration: 1500 });
    });
    expect(screen.getByRole('status')).toHaveTextContent('临时提示');

    act(() => {
      vi.advanceTimersByTime(1600);
    });
    expect(screen.queryByRole('status')).not.toBeInTheDocument();

    vi.useRealTimers();
  });

  it('同 ID 替换后重新计时，旧版本关闭回调不能移除新通知', () => {
    vi.useFakeTimers();
    render(<ToastContainer />);

    act(() => {
      toast.info('旧提示', { id: 'replaceable', duration: 1000 });
    });
    const oldRevision = useToastStore.getState().toasts[0].revision;
    act(() => vi.advanceTimersByTime(900));

    act(() => {
      toast.info('新提示', { id: 'replaceable', duration: 1000 });
    });
    expect(useToastStore.getState().toasts[0].revision).not.toBe(oldRevision);

    act(() => {
      vi.advanceTimersByTime(200);
      useToastStore.getState().dismissToast('replaceable', oldRevision);
    });
    expect(screen.getByRole('status')).toHaveTextContent('新提示');

    act(() => vi.advanceTimersByTime(800));
    expect(screen.queryByRole('status')).not.toBeInTheDocument();
    vi.useRealTimers();
  });
});
