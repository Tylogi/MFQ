/** Display the model-directory browser and registration dialog, reusing the directory controller’s operation state. */
import { useTranslation } from 'react-i18next';
import { Dialog } from '../../shared/ui/Dialog';
import { Icon } from '../../app/display';
import type { useModelCatalog } from './useModelCatalog';
/** Render the server folder picker and restore focus to its trigger when closed. */
export function ModelDirectoryDialog({ catalog }: { catalog: ReturnType<typeof useModelCatalog> }) {
  const { t } = useTranslation();
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
      title={t('models:modelDirectoryDialog.chooseModelFolder')}
      description={t('models:modelDirectoryDialog.browseFoldersOnTheMfqServerHost')}
      closeLabel={t('common:close')}
      className="model-browser-dialog"
      returnFocusRef={modelBrowserTriggerRef}
    >
      <form className="model-browser-location" onSubmit={jumpToModelDirectory}>
        <button
          disabled={busy || !modelBrowser.current_id}
          onClick={() => void openModelDirectory(modelBrowser.parent_id)}
          type="button"
        >
          {t('models:modelDirectoryDialog.up')}
        </button>
        <input
          aria-label={t('models:modelDirectoryDialog.currentDirectory')}
          onChange={(event) => setModelDirectoryPath(event.target.value)}
          placeholder={t('models:modelDirectoryDialog.enterAFullDirectoryOnTheServer')}
          spellCheck={false}
          value={modelDirectoryPath}
        />
        <button disabled={busy || !modelDirectoryPath.trim()} type="submit">
          {t('models:modelDirectoryDialog.go')}
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
          <p>{t('models:modelDirectoryDialog.thisFolderHasNoSubfolders')}</p>
        )}
      </div>
      <footer>
        <button onClick={() => setModelBrowserOpen(false)} type="button">
          {t('common:cancel')}
        </button>
        <button
          className="primary"
          disabled={busy || !modelBrowser.current_id}
          onClick={() => void registerCurrentModelDirectory()}
          type="button"
        >
          {t('models:modelDirectoryDialog.useThisFolder')}
        </button>
      </footer>
    </Dialog>
  );
}
