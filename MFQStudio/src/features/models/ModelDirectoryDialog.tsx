/** 展示模型目录浏览和注册弹窗，复用目录控制器的操作状态。 */
import { Dialog } from '../../shared/ui/Dialog';
import { Icon } from '../../app/display';
import { useSettings } from '../settings/SettingsProvider';
import type { useModelCatalog } from './useModelCatalog';

/** 渲染服务器文件夹选择器，并在关闭时恢复触发器焦点。 */
export function ModelDirectoryDialog({ catalog }: { catalog: ReturnType<typeof useModelCatalog> }) {
  const { tr } = useSettings();
  const {
    modelBrowser,
    modelBrowserOpen,
    setModelBrowserOpen,
    modelBrowserTriggerRef,
    busy,
    jumpToModelDirectory,
    openModelDirectory,
    modelDirectoryPath,
    setModelDirectoryPath,
    registerCurrentModelDirectory,
  } = catalog;
  if (!modelBrowser) return null;
  return (
    <Dialog
      open={modelBrowserOpen}
      onOpenChange={setModelBrowserOpen}
      title={tr('选择模型文件夹', 'Choose model folder')}
      description={tr(
        '浏览 MFQ Server 所在设备上的文件夹。',
        'Browse folders on the MFQ Server host.',
      )}
      closeLabel={tr('关闭', 'Close')}
      className="model-browser-dialog"
      returnFocusRef={modelBrowserTriggerRef}
    >
      <form className="model-browser-location" onSubmit={jumpToModelDirectory}>
        <button
          disabled={busy || !modelBrowser.current_id}
          onClick={() => void openModelDirectory(modelBrowser.parent_id)}
          type="button"
        >
          {tr('上一级', 'Up')}
        </button>
        <input
          aria-label={tr('当前目录', 'Current directory')}
          onChange={(event) => setModelDirectoryPath(event.target.value)}
          placeholder={tr('输入服务器上的完整目录', 'Enter a full directory on the server')}
          spellCheck={false}
          value={modelDirectoryPath}
        />
        <button disabled={busy || !modelDirectoryPath.trim()} type="submit">
          {tr('前往', 'Go')}
        </button>
        {modelBrowser.current_id && <span>{modelBrowser.model_file_count} MFQ</span>}
      </form>
      <div className="model-browser-list">
        {modelBrowser.data.map((directory) => (
          <button
            disabled={busy}
            key={directory.id}
            onClick={() => void openModelDirectory(directory.id)}
            type="button"
          >
            <Icon name="folder" />
            <span>{directory.name}</span>
            {directory.model_file_count > 0 && <b>{directory.model_file_count} MFQ</b>}
          </button>
        ))}
        {modelBrowser.data.length === 0 && (
          <p>{tr('这个文件夹中没有子文件夹。', 'This folder has no subfolders.')}</p>
        )}
      </div>
      <footer>
        <button onClick={() => setModelBrowserOpen(false)} type="button">
          {tr('取消', 'Cancel')}
        </button>
        <button
          className="primary"
          disabled={busy || !modelBrowser.current_id}
          onClick={() => void registerCurrentModelDirectory()}
          type="button"
        >
          {tr('使用此文件夹', 'Use this folder')}
        </button>
      </footer>
    </Dialog>
  );
}
