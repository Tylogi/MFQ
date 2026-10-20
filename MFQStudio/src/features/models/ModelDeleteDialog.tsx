import { useSettings } from '../settings/SettingsProvider';
import { Dialog } from '../../shared/ui/Dialog';
import { formatNumber } from '../../app/formatters';
import type { useModelCatalog } from './useModelCatalog';
import './model-delete.css';

export function ModelDeleteDialog({ catalog }: { catalog: ReturnType<typeof useModelCatalog> }) {
  const { tr } = useSettings();
  const { modelDeletion: plan, busy, cancelModelDeletion, confirmModelDeletion, deletionTriggerRef } = catalog;
  if (!plan) return null;
  return <Dialog open onOpenChange={open => { if (!open) cancelModelDeletion(); }}
    title={tr('删除模型', 'Delete model')} closeLabel={tr('取消', 'Cancel')}
    description={tr('将永久删除以下实际文件，无法撤销。不会删除同目录中的其它模型或聊天记录。',
      'The files below will be permanently deleted. This cannot be undone. Other models and chat history will not be deleted.')}
    className="model-delete-dialog" returnFocusRef={deletionTriggerRef}>
    <div className="model-delete-details">
      <strong>{plan.name}</strong>
      <span>{tr(`${plan.files.length} 个文件`, `${plan.files.length} files`)} · {formatNumber(plan.total_bytes / 2 ** 30, 2)} GiB</span>
      <ul aria-label={tr('将删除的文件', 'Files to delete')}>{plan.files.map(path => <li key={path}><code>{path}</code></li>)}</ul>
    </div>
    <footer className="model-delete-actions">
      <button disabled={busy} onClick={cancelModelDeletion} type="button">{tr('取消', 'Cancel')}</button>
      <button className="danger" disabled={busy} onClick={() => void confirmModelDeletion()} type="button">
        {busy ? tr('正在删除…', 'Deleting…') : tr('确认删除', 'Confirm deletion')}
      </button>
    </footer>
  </Dialog>;
}
