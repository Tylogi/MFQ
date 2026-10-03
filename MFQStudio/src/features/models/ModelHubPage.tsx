import { type CSSProperties, useEffect, useRef, useState } from 'react';
import { jobsApi } from '../../shared/api/resources/jobs';
import type { JobKindResource, JobResource } from '../../shared/api/types';
import { useRuntime } from '../../app/RuntimeProvider';
import { errorMessage } from '../../app/formatters';
import { toast } from '../../stores/toastStore';
import { useSettings } from '../settings/SettingsProvider';
import { Icon } from '../../app/display';
import { useJobStore } from '../../stores/jobStore';
import { ModelBrowser, type DownloadOrigin, type ModelBrowserTab } from './ModelBrowser';
import { DownloadQueue, downloadProgress, isActiveDownload, isModelDownload } from './DownloadQueue';

export function ModelHubPage() {
  const { tr } = useSettings();
  const { addJob } = useRuntime();
  const jobs = useJobStore((state) => state.jobs);
  const downloads = jobs.filter(isModelDownload);
  const active = downloads.filter(isActiveDownload);
  const progress = active.length ? active.reduce((sum, job) => sum + downloadProgress(job), 0) / active.length : 0;
  const [tab, setTab] = useState<ModelBrowserTab>('official');
  const circle = useRef<HTMLButtonElement>(null);
  const page = useRef<HTMLDivElement>(null);
  const [flight, setFlight] = useState<{ id: string; origin: DownloadOrigin; dx: number; dy: number } | null>(null);
  const [jobKinds, setJobKinds] = useState<JobKindResource[]>([]);

  useEffect(() => {
    let active = true;
    void jobsApi
      .jobKinds()
      .then((items) => {
        if (active) setJobKinds(items);
      })
      .catch((cause) => {
        if (active) toast.error(errorMessage(cause));
      });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (!flight) return;
    const timer = setTimeout(() => setFlight(null), 750);
    return () => clearTimeout(timer);
  }, [flight]);

  function trackDownload(job: JobResource, origin: DownloadOrigin) {
    addJob(job);
    toast.success(tr('下载任务已提交', 'Download job submitted'));
    const rect = circle.current?.getBoundingClientRect();
    if (rect) setFlight({ id: job.id, origin, dx: rect.left + rect.width / 2 - origin.x, dy: rect.top + rect.height / 2 - origin.y });
  }

  return (
    <div className="model-download-page" ref={page}>
      <ModelBrowser
        jobKinds={jobKinds}
        onError={(message) => toast.error(message)}
        onJobCreated={trackDownload}
        tab={tab}
        onTabChange={setTab}
        downloadQueue={<DownloadQueue jobs={downloads} onJobCreated={trackDownload} />}
        tr={tr}
      />
      <button ref={circle} className={`download-circle${flight ? ' receiving' : ''}`} type="button"
        aria-label={tr('下载队列', 'Download queue')} title={tr('下载队列', 'Download queue')}
        onClick={() => {
          setTab('downloads');
          page.current?.scrollIntoView({ block: 'start', behavior: 'smooth' });
        }}>
        <svg className="download-circle-ring" viewBox="0 0 56 56" aria-hidden="true">
          <circle cx="28" cy="28" r="24" />
          {active.length > 0 && <circle className="download-ring-progress" cx="28" cy="28" r="24" pathLength="100" strokeDasharray={`${progress * 100} 100`} />}
        </svg>
        <Icon name="download" size={22} />
        {active.length > 0 && <span className="download-circle-count">{active.length}</span>}
      </button>
      {flight && <span className="download-flight" key={flight.id} aria-hidden="true" style={{
        left: flight.origin.x - 14, top: flight.origin.y - 14,
        '--flight-x': `${flight.dx}px`, '--flight-y': `${flight.dy}px`,
      } as CSSProperties}><Icon name="download" size={18} /></span>}
    </div>
  );
}
