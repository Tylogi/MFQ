/** 验证顶层故障视图可脱离路由和设置上下文独立渲染。 */
import { fireEvent, render, screen } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import { FailureView } from './FailurePage';

it('显示完整故障信息并执行恢复操作', () => {
  const onRetry = vi.fn();
  const onLeave = vi.fn();
  render(
    <FailureView
      code="APP / 03"
      description="应用暂时无法显示"
      detail="render failed"
      detailLabel="查看错误详情"
      kind="render"
      leaveLabel="返回概览"
      onLeave={onLeave}
      onRetry={onRetry}
      retryLabel="重新载入"
      title="界面遇到错误"
    />,
  );

  expect(screen.getByRole('heading', { name: '界面遇到错误' })).toBeTruthy();
  fireEvent.click(screen.getByText('查看错误详情'));
  expect(screen.getByText('render failed')).toBeTruthy();
  fireEvent.click(screen.getByRole('button', { name: '重新载入' }));
  fireEvent.click(screen.getByRole('button', { name: '返回概览' }));
  expect(onRetry).toHaveBeenCalledOnce();
  expect(onLeave).toHaveBeenCalledOnce();
});
