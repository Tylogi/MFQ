/** Verify keyboard and screen-reader behavior for modals, tooltips, and immediate settings switches. */
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
  // jsdom has no layout observer; this group checks only focus, keyboard, and accessibility semantics.
  vi.stubGlobal(
    'ResizeObserver',
    class {
      observe = vi.fn();
      unobserve = vi.fn();
      disconnect = vi.fn();
    },
  );
});

/** Simulate two business entry points sharing a controlled dialog with an explicit focus-return target. */
function DialogFixture({ explicitTarget = false }: { explicitTarget?: boolean }) {
  const [open, setOpen] = useState(false);
  const returnFocusRef = useRef<HTMLButtonElement | null>(null);
  return (
    <>
      <button type="button" ref={returnFocusRef} onClick={() => setOpen(true)}>
        Entry one
      </button>
      <button type="button" onClick={() => setOpen(true)}>
        Entry two
      </button>
      <Dialog
        open={open}
        onOpenChange={setOpen}
        title="Model directory"
        description="Choose a server directory"
        closeLabel="Close directory"
        returnFocusRef={explicitTarget ? returnFocusRef : undefined}
      >
        <input aria-label="Directory path" />
        <button type="button" onClick={() => setOpen(false)}>
          Confirm directory
        </button>
      </Dialog>
    </>
  );
}

describe('Dialog', () => {
  it('verifies primitives test behavior 1', async () => {
    const user = userEvent.setup();
    render(<DialogFixture />);
    for (const name of ['Entry one', 'Entry two']) {
      const trigger = screen.getByRole('button', { name });
      await user.click(trigger);
      const dialog = screen.getByRole('dialog', { name: 'Model directory' });
      expect(dialog).toHaveAccessibleDescription('Choose a server directory');
      expect(dialog).toContainElement(document.activeElement as HTMLElement);
      await user.keyboard('{Escape}');
      await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
      await waitFor(() => expect(trigger).toHaveFocus());
    }
  });

  it('verifies primitives test behavior 2', async () => {
    const user = userEvent.setup();
    render(<DialogFixture />);
    await user.click(screen.getByRole('button', { name: 'Entry one' }));
    expect(screen.getByRole('button', { name: 'Close directory' })).toHaveFocus();
    await user.tab({ shift: true });
    expect(screen.getByRole('button', { name: 'Confirm directory' })).toHaveFocus();
    await user.tab();
    expect(screen.getByRole('button', { name: 'Close directory' })).toHaveFocus();
  });

  it('verifies primitives test behavior 3', async () => {
    const user = userEvent.setup();
    render(<DialogFixture explicitTarget />);
    await user.click(screen.getByRole('button', { name: 'Entry two' }));
    await user.click(screen.getByRole('button', { name: 'Close directory' }));
    await waitFor(() => expect(screen.getByRole('button', { name: 'Entry one' })).toHaveFocus());
  });
});
describe('Switch', () => {
  it('verifies primitives test behavior 4', async () => {
    const user = userEvent.setup();
    const change = vi.fn();
    const submit = vi.fn((event: React.FormEvent) => event.preventDefault());
    const view = render(
      <form onSubmit={submit}>
        <Switch label="Follow automatically" checked={false} onCheckedChange={change} />
      </form>,
    );
    await user.tab();
    await user.keyboard(' ');
    expect(change).toHaveBeenCalledWith(true);
    expect(submit).not.toHaveBeenCalled();
    view.rerender(<Switch label="Follow automatically" checked onCheckedChange={change} />);
    expect(screen.getByRole('switch', { name: 'Follow automatically' })).toBeChecked();
  });

  it('verifies primitives test behavior 5', async () => {
    const user = userEvent.setup();
    const change = vi.fn();
    render(<Switch label="Follow automatically" checked={false} onCheckedChange={change} disabled />);
    await user.click(screen.getByRole('switch', { name: 'Follow automatically' }));
    expect(change).not.toHaveBeenCalled();
  });
});

describe('Tooltip', () => {
  it('verifies primitives test behavior 6', async () => {
    const user = userEvent.setup();
    render(
      <TooltipProvider delayDuration={0}>
        <Tooltip content="Regenerate response">
          <button type="button" aria-label="Regenerate">
            R
          </button>
        </Tooltip>
      </TooltipProvider>,
    );
    await user.tab();
    expect(await screen.findByRole('tooltip')).toHaveTextContent('Regenerate response');
    expect(screen.getByRole('button', { name: 'Regenerate' })).toHaveFocus();
    await user.keyboard('{Escape}');
    await waitFor(() => expect(screen.queryByRole('tooltip')).not.toBeInTheDocument());
  });
});

describe('Toast', () => {
  beforeEach(() => {
    useToastStore.getState().clearToasts();
  });

  it('verifies primitives test behavior 7', async () => {
    const user = userEvent.setup();
    render(<ToastContainer />);

    expect(screen.queryByRole('region', { name: 'Notifications' })).not.toBeInTheDocument();

    toast.error('Model loading failed', { title: 'Error' });
    toast.success('Configuration saved');

    expect(await screen.findByRole('alert')).toHaveTextContent('Model loading failed');
    expect(screen.getByText('Error')).toBeInTheDocument();
    expect(screen.getByRole('status')).toHaveTextContent('Configuration saved');

    const closeButtons = screen.getAllByRole('button', { name: '关闭通知' });
    expect(closeButtons).toHaveLength(2);

    await user.click(closeButtons[0]);
    await waitFor(() => expect(screen.queryByRole('alert')).not.toBeInTheDocument());
    expect(screen.getByRole('status')).toHaveTextContent('Configuration saved');
  });

  it('verifies primitives test behavior 8', async () => {
    vi.useFakeTimers();
    render(<ToastContainer />);

    act(() => {
      toast.info('Temporary notice', { duration: 1500 });
    });
    expect(screen.getByRole('status')).toHaveTextContent('Temporary notice');

    act(() => {
      vi.advanceTimersByTime(1600);
    });
    expect(screen.queryByRole('status')).not.toBeInTheDocument();

    vi.useRealTimers();
  });

  it('verifies primitives test behavior 9', () => {
    vi.useFakeTimers();
    render(<ToastContainer />);

    act(() => {
      toast.info('Old notice', { id: 'replaceable', duration: 1000 });
    });
    const oldRevision = useToastStore.getState().toasts[0].revision;
    act(() => vi.advanceTimersByTime(900));

    act(() => {
      toast.info('New notice', { id: 'replaceable', duration: 1000 });
    });
    expect(useToastStore.getState().toasts[0].revision).not.toBe(oldRevision);

    act(() => {
      vi.advanceTimersByTime(200);
      useToastStore.getState().dismissToast('replaceable', oldRevision);
    });
    expect(screen.getByRole('status')).toHaveTextContent('New notice');

    act(() => vi.advanceTimersByTime(800));
    expect(screen.queryByRole('status')).not.toBeInTheDocument();
    vi.useRealTimers();
  });
});
