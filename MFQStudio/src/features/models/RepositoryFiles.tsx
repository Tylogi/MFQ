/** Browse repository folders and select files for download. */
import { i18n } from '../../i18n';
import type { TFunction } from 'i18next';
import { useState } from 'react';
import { Icon } from '../../app/display';
import type { HubModelInfo } from '../../shared/api/types';
import type { DownloadOrigin } from './ModelBrowser';

type HubModelFile = HubModelInfo['files'][number];

/** Browse repository folders and select files for download. */
export function RepositoryFiles({ files, disabled, onDownload, t }: {
  files: HubModelFile[];
  disabled: boolean;
  onDownload(files: HubModelFile[], label: string, origin: DownloadOrigin): void;
  t: TFunction;
}) {
  const [folder, setFolder] = useState('');
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const contained = files.filter((file) => file.name.startsWith(folder));
  const folders = Array.from(new Set(contained.map((file) => file.name.slice(folder.length))
    .filter((name) => name.includes('/')).map((name) => name.split('/')[0]))).sort();
  const leaves = contained.filter((file) => !file.name.slice(folder.length).includes('/'));
  const chosen = files.filter((file) => selected.has(file.name));
  function submit(event: React.MouseEvent<HTMLButtonElement>, items: HubModelFile[], label: string) {
    const rect = event.currentTarget.getBoundingClientRect();
    onDownload(items, label, { x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 });
  }
  return <details className="repository-files">
    <summary>{t('models:repositoryFiles.filesAndFolders')} · {files.length}</summary>
    <div className="repository-file-toolbar">
      <button disabled={!folder} onClick={() => setFolder(folder.replace(/[^/]+\/$/, ''))} type="button">{t('models:repositoryFiles.up')}</button>
      <code>/{folder}</code>
      <button disabled={disabled || !contained.length} onClick={(event) => submit(event, contained, folder.replace(/\/$/, '') || 'repository')} type="button">{t('models:repositoryFiles.downloadFolder')}</button>
    </div>
    <div className="repository-file-list">
      {folders.map((name) => <button key={name} className="repository-folder" type="button" onClick={() => setFolder(`${folder}${name}/`)}>
        <Icon name="folder" size={17} /><span>{name}/</span><small>{contained.filter((file) => file.name.startsWith(`${folder}${name}/`)).length} {t('models:repositoryFiles.files')}</small>
      </button>)}
      {leaves.map((file) => <label key={file.name}>
        <input type="checkbox" checked={selected.has(file.name)} onChange={(event) => setSelected((current) => {
          const next = new Set(current);
          if (event.target.checked) next.add(file.name); else next.delete(file.name);
          return next;
        })} />
        <span>{file.name.slice(folder.length)}</span><small>{(file.byte_size / 2 ** 20).toLocaleString(i18n.resolvedLanguage, { maximumFractionDigits: 1 })} MiB</small>
      </label>)}
    </div>
    <button disabled={disabled || !chosen.length} onClick={(event) => submit(event, chosen, 'selected-files')} type="button">
      {t('models:repositoryFiles.downloadSelectedFiles', { count: chosen.length })}
    </button>
  </details>;
}
