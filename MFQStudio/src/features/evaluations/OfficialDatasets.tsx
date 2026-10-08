import { Icon } from '../../app/display';
import type { DatasetResource, OfficialDataset, JobResource } from '../../shared/api/types';

export function OfficialDatasets({ catalog, datasets, jobs, busy, readiness, submit, tr }: {
  catalog: OfficialDataset[]; datasets: DatasetResource[]; jobs: JobResource[]; busy: boolean;
  readiness?: Record<string, { reason?: string }>;
  submit: (kind: string, payload: Record<string, unknown>) => Promise<void>; tr: (zh: string, en: string) => string;
}) {
  return <section className="tm-panel evaluation-library">
    <div className="panel-heading"><div><h2>{tr('官方测评集合', 'Official test collections')}</h2><p>{tr('固定版本 · 完整文件 SHA256 · 运行前再次校验', 'Pinned revisions · whole-file SHA256 · reverified before evaluation')}</p></div></div>
    <div className="evaluation-library-grid">{catalog.map((spec) => {
      const local = datasets.find((item) => item.sha256 === spec.sha256 && item.byte_size === spec.byte_size);
      const needsScorer = !!local && readiness?.[spec.id]?.reason === 'official_scorer_missing';
      const job = jobs.find((item) => item.kind === 'dataset.download' && item.payload.dataset === spec.id && ['queued', 'running', 'cancelling'].includes(item.status));
      return <article key={spec.id}>
        <header><Icon name="folder" size={18} /><div><h3>{spec.name}</h3><a href={spec.origin === 'github' ? `https://github.com/${spec.repository}/tree/${spec.revision}` : `https://huggingface.co/datasets/${spec.repository}/tree/${spec.revision}`} target="_blank" rel="noreferrer">{spec.repository}</a></div></header>
        <dl><div><dt>{tr('源划分', 'Source split')}</dt><dd>{spec.split || 'test'}</dd></div><div><dt>{tr('样本', 'Samples')}</dt><dd>{spec.rows.toLocaleString()}</dd></div><div><dt>{tr('大小', 'Size')}</dt><dd>{(spec.byte_size / 2 ** 20).toFixed(2)} MiB</dd></div></dl>
        <details><summary>{tr('版本与完整哈希', 'Revision & full hash')}</summary><p>{tr('版本', 'Revision')}<code>{spec.revision}</code></p><p>SHA256<code>{spec.sha256}</code></p><p>{tr('文件', 'File')}<code>{spec.filename}</code></p><p>{tr('许可', 'License')}<span>{spec.license}</span></p>{local && <p>{tr('本地位置', 'Local path')}<code>{local.artifact_uri}</code></p>}</details>
        <footer><span>{job ? tr(`下载中 ${(job.progress * 100).toFixed(0)}%`, `Downloading ${(job.progress * 100).toFixed(0)}%`) : local ? tr('已下载', 'Downloaded') : tr('未下载', 'Not downloaded')}</span>
          <button type="button" className="evaluation-run" disabled={busy || !!job || (!!local && !needsScorer)} onClick={() => void submit('dataset.download', { dataset: spec.id })}><Icon name={local && !needsScorer ? 'check' : 'download'} size={14} />{needsScorer ? tr('校验并安装评分', 'Verify & install scoring') : local ? tr('数据已就绪', 'Data ready') : tr('下载集合', 'Download collection')}</button></footer>
        {job && <progress aria-label={tr('集合下载进度', 'Collection download progress')} value={job.progress} max={1} />}
      </article>;
    })}</div>
    {!catalog.length && <div className="inline-empty">{tr('正在读取官方测评目录', 'Loading the official test catalog')}</div>}
  </section>;
}
