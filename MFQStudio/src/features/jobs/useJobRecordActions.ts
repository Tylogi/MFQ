/** 量化任务取消、重试、记录清理及产物删除操作。 */
import { useState, type Dispatch, type SetStateAction } from 'react';
import type { JobResource, RuntimeLogEntry } from '../../shared/api/types';
import { jobsApi } from '../../shared/api/resources/jobs';
import { modelsApi } from '../../shared/api/resources/models';
import { studioConfirm } from '../../studio';
import { errorMessage } from '../../app/formatters';
import { useSettings } from '../settings/SettingsProvider';
import { useRuntime } from '../../app/RuntimeProvider';
import { useJobStore } from '../../stores/jobStore';
import { toast } from '../../stores/toastStore';
import { isTerminalJob } from './jobSchema';

/** 操作选中任务及任务历史，清理后同步共享运行时。 */
export function useJobRecordActions(
  selectedJobId: string | null,
  setSelectedJobId: Dispatch<SetStateAction<string | null>>,
  selectedJob: JobResource | null,
  busy: boolean,
  setBusy: Dispatch<SetStateAction<boolean>>,
  setJobLogs: Dispatch<SetStateAction<RuntimeLogEntry[]>>,
) {
  const { tr } = useSettings();
  const { refreshRuntime } = useRuntime();
  const addJob = useJobStore((state) => state.addJob);
  const [jobCleanupBusy, setJobCleanupBusy] = useState(false);

  /** 请求取消选中的后台任务。 */
  async function cancelSelectedJob() {
    if (!selectedJobId) return;
    try {
      addJob(await jobsApi.cancelJob(selectedJobId));
    } catch (cause) {
      toast.error(errorMessage(cause));
    }
  }

  /** 为失败任务创建重试记录。 */
  async function retrySelectedJob() {
    if (!selectedJob || busy) return;
    setBusy(true);
    try {
      const retried = await jobsApi.retryJob(selectedJob.id);
      addJob(retried);
      setSelectedJobId(retried.id);
    } catch (cause) {
      toast.error(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  /** 移除终态任务记录并清理当前选择。 */
  async function deleteJobRecord(id: string) {
    if (jobCleanupBusy) return;
    setJobCleanupBusy(true);
    try {
      await jobsApi.deleteJob(id);
      await refreshRuntime(false);
      if (selectedJobId === id) {
        setSelectedJobId(null);
        setJobLogs([]);
      }
    } catch (cause) {
      toast.error(errorMessage(cause));
    } finally {
      setJobCleanupBusy(false);
    }
  }

  /** 清理终态任务历史并同步共享运行时。 */
  async function clearCompletedJobRecords() {
    if (jobCleanupBusy) return;
    setJobCleanupBusy(true);
    try {
      await jobsApi.clearCompletedJobs();
      await refreshRuntime(false);
      if (selectedJob && isTerminalJob(selectedJob)) {
        setSelectedJobId(null);
        setJobLogs([]);
      }
    } catch (cause) {
      toast.error(errorMessage(cause));
    } finally {
      setJobCleanupBusy(false);
    }
  }

  /** 确认后删除选中任务的本地产物。 */
  async function removeSelectedArtifact() {
    const uri = String(selectedJob?.result?.artifact || '');
    if (!uri.startsWith('workspace://') || busy) return;
    if (!(await studioConfirm(tr('删除这个本地产物？', 'Delete this local artifact?')))) return;
    setBusy(true);
    try {
      await modelsApi.removeWorkspaceArtifact(uri);
      await refreshRuntime(false);
    } catch (cause) {
      toast.error(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }
  return {
    jobCleanupBusy, cancelSelectedJob, retrySelectedJob, deleteJobRecord,
    clearCompletedJobRecords, removeSelectedArtifact,
  };
}
