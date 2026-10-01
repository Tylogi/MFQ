/** 模型仓库页面组合官方目录、第三方仓库解析与可续传下载任务。 */
import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router';
import { jobsApi } from '../../shared/api/resources/jobs';
import type { JobKindResource, JobResource } from '../../shared/api/types';
import { useRuntime } from '../../app/RuntimeProvider';
import { errorMessage } from '../../app/formatters';
import { toast } from '../../stores/toastStore';
import { useSettings } from '../settings/SettingsProvider';
import { ModelBrowser } from './ModelBrowser';

/** 按需读取下载任务能力，并把新任务交给共享运行时持续跟踪。 */
export function ModelHubPage() {
  const { tr } = useSettings();
  const { addJob } = useRuntime();
  const navigate = useNavigate();
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

  function trackDownload(job: JobResource) {
    addJob(job);
    toast.success(tr('下载任务已提交', 'Download job submitted'));
    navigate('/quantization', { state: { jobId: job.id } });
  }

  return (
    <ModelBrowser
      jobKinds={jobKinds}
      onError={(message) => toast.error(message)}
      onJobCreated={trackDownload}
      tr={tr}
    />
  );
}
