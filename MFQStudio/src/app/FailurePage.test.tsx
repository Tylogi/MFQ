/** Verify that the top-level failure view renders independently of router and settings context. */
import { fireEvent, render, screen } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import { FailureView } from './FailurePage';

it('verifies FailurePage test behavior 1', () => {
  const onRetry = vi.fn();
  const onLeave = vi.fn();
  render(
    <FailureView
      code="APP / 03"
      description="The application is temporarily unavailable"
      detail="render failed"
      detailLabel="View error details"
      kind="render"
      leaveLabel="Back to overview"
      onLeave={onLeave}
      onRetry={onRetry}
      retryLabel="Reload"
      title="The interface encountered an error"
    />,
  );

  expect(screen.getByRole('heading', { name: 'The interface encountered an error' })).toBeTruthy();
  fireEvent.click(screen.getByText('View error details'));
  expect(screen.getByText('render failed')).toBeTruthy();
  fireEvent.click(screen.getByRole('button', { name: 'Reload' }));
  fireEvent.click(screen.getByRole('button', { name: 'Back to overview' }));
  expect(onRetry).toHaveBeenCalledOnce();
  expect(onLeave).toHaveBeenCalledOnce();
});
