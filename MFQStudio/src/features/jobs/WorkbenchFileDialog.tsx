import { useEffect, useState } from 'react';
import { FileIcon, FolderIcon, ArrowUpIcon } from '@phosphor-icons/react';
import { quantizationApi, type QuantizationFiles } from '../../shared/api/resources/quantization';
import { Dialog } from '../../shared/ui/Dialog';
import { useSettings } from '../settings/SettingsProvider';

interface Props {
  initialPath: string;
  directories: boolean;
  allowFiles?: boolean;
  title: string;
  onSelect: (path: string) => void;
  onClose: () => void;
}

export function WorkbenchFileDialog({ initialPath, directories, allowFiles = false, title, onSelect, onClose }: Props) {
  const { tr } = useSettings();
  const [path, setPath] = useState(initialPath);
  const [directory, setDirectory] = useState(initialPath);
  const [files, setFiles] = useState<QuantizationFiles | null>(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    const controller = new AbortController();
    setLoading(true); setError(''); setFiles(null);
    void quantizationApi.files(directory, controller.signal).then((result) => {
      if (controller.signal.aborted) return;
      setFiles(result); setPath(result.path);
    }).catch((cause) => { if (!controller.signal.aborted) setError(String(cause)); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [directory]);
  return <Dialog open onOpenChange={(open) => { if (!open) onClose(); }} title={title} closeLabel={tr('关闭', 'Close')} className="workbench-file-dialog">
    <form className="qw-path-row" onSubmit={(event) => { event.preventDefault(); setDirectory(path); }}>
      <button type="button" disabled={loading || !(files?.parent || directory.replace(/\/+$/, '').replace(/\/[^/]+$/, '') || (directory !== '/' ? '/' : ''))} onClick={() => setDirectory(files?.parent || directory.replace(/\/+$/, '').replace(/\/[^/]+$/, '') || '/')} aria-label={tr('上一级', 'Parent directory')}><ArrowUpIcon size={16} /></button>
      <input aria-label={tr('目录路径', 'Directory path')} value={path} onChange={(event) => setPath(event.target.value)} />
      <button type="submit">{tr('前往', 'Go')}</button>
    </form>
    {error && <p role="alert" className="qw-error">{error}</p>}
    <div className="qw-file-list" aria-busy={loading}>
      {loading ? <p>{tr('读取目录…', 'Reading directory…')}</p> : files?.data.map((file) => <button key={file.path} type="button" disabled={directories && !allowFiles && !file.directory} onClick={() => file.directory ? setDirectory(file.path) : onSelect(file.path)}>
        {file.directory ? <FolderIcon size={19} /> : <FileIcon size={19} />}<span>{file.name}</span>
        {!file.directory && file.byte_size !== null && <small>{(file.byte_size / 1048576).toFixed(1)} MiB</small>}
      </button>)}
      {!loading && !error && files?.data.length === 0 && <p>{tr('此目录为空', 'This directory is empty')}</p>}
    </div>
    {directories && <footer><button type="button" disabled={loading || !!error || files?.path !== directory} onClick={() => { if (files) onSelect(files.path); }}>{tr('使用此目录', 'Use this directory')}</button></footer>}
  </Dialog>;
}
