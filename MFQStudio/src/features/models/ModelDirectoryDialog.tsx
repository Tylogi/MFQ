/** 展示模型目录浏览和注册弹窗，复用目录控制器的操作状态。 */
import { Dialog } from '../../shared/ui/Dialog';
import { Icon } from '../../app/display';
import { useSettings } from '../settings/SettingsProvider';
import type { useModelCatalog } from './useModelCatalog';
import { formatBytes } from '../../app/formatters';

/** 渲染服务器文件夹选择器，并在关闭时恢复触发器焦点。 */
export function ModelDirectoryDialog({ catalog }: { catalog: ReturnType<typeof useModelCatalog> }) {
  const { tr } = useSettings();
  const {
    modelBrowser,
    modelBrowserOpen,
    modelBrowserError,
    modelBrowserLoading,
    modelFilesMode,
    setModelBrowserOpen,
    modelBrowserTriggerRef,
    busy,
    jumpToModelDirectory,
    openModelDirectory,
    modelDirectoryPath,
    setModelDirectoryPath,
    registerCurrentModelDirectory,
    openCurrentDirectoryInFinder,
  } = catalog;
  return (
    <Dialog
      open={modelBrowserOpen}
      onOpenChange={setModelBrowserOpen}
      title={modelFilesMode ? tr('模型文件', 'Model files') : tr('选择模型文件夹', 'Choose model folder')}
      description={modelFilesMode ? tr('模型所在的实际目录。', 'The directory containing this model’s files.') : tr(
        '浏览 MFQ Server 所在设备上的文件夹。',
        'Browse folders on the MFQ Server host.',
      )}
      closeLabel={tr('关闭', 'Close')}
      className="model-browser-dialog"
      returnFocusRef={modelBrowserTriggerRef}
    >
      <form className="model-browser-location" onSubmit={jumpToModelDirectory}>
        <button
          disabled={busy || !modelBrowser?.parent_id}
          onClick={() => void openModelDirectory(modelBrowser?.parent_id)}
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
        {modelBrowser?.current_id && <span>{modelBrowser.model_file_count} {tr('个模型', 'models')}</span>}
      </form>
      <div className="model-browser-list">
        {modelBrowserLoading && <p role="status">{tr('正在读取目录', 'Loading folder')}</p>}
        {modelBrowserError && <p role="alert">{modelBrowserError}</p>}
        {modelBrowser?.data.map((directory) => (
          <button
            disabled={busy}
            key={directory.id}
            onClick={() => void openModelDirectory(directory.id)}
            type="button"
          >
            <Icon name="folder" />
            <span>{directory.name}</span>
            {directory.model_file_count > 0 && <b>{directory.model_file_count} {tr('个模型', 'models')}</b>}
          </button>
        ))}
        {modelBrowser?.files?.map((file) => <div className="model-browser-file" key={file.name}>
          <Icon name="file" />
          <span title={file.name}>{file.name}</span>
          <small>{file.byte_size === 0 ? '0 B' : formatBytes(file.byte_size).replace(/\b(KB|MB|GB|TB)\b/g, (unit) => `${unit[0]}iB`)}</small>
        </div>)}
        {modelBrowser && modelBrowser.data.length === 0 && !modelBrowser.files?.length && !modelBrowserLoading && !modelBrowserError && (
          <p>{tr('这个文件夹中没有子文件夹或 MFQ 文件。', 'This folder has no subfolders or MFQ files.')}</p>
        )}
      </div>
      <footer>
        <button className="model-browser-open-finder" disabled={busy || !modelBrowser?.current_id || !modelBrowser.can_open_in_finder || Boolean(modelBrowserError)}
          onClick={() => void openCurrentDirectoryInFinder()} type="button">
          <Icon name="folder" size={13} />{tr('在访达中打开', 'Open in Finder')}
        </button>
        <button onClick={() => setModelBrowserOpen(false)} type="button">
          {tr('取消', 'Cancel')}
        </button>
        {!modelFilesMode && <button
          className="primary"
          disabled={busy || !modelBrowser?.current_id || Boolean(modelBrowserError)}
          onClick={() => void registerCurrentModelDirectory()}
          type="button"
        >
          {tr('使用此文件夹', 'Use this folder')}
        </button>}
      </footer>
    </Dialog>
  );
}
