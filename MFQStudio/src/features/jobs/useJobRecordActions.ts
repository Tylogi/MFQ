/** Cancel and retry quantization jobs, clear records, and delete artifacts. */
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
/** Operate on the selected job and job history, synchronizing shared runtime state after cleanup. */
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
/** Request cancellation of the selected background job. */
  async function cancelSelectedJob() {
    if (!selectedJobId) return;
    try {
      addJob(await jobsApi.cancelJob(selectedJobId));
    } catch (cause) {
      toast.error(errorMessage(cause));
    }
  }
/** Create a retry record for a failed job. */
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
/** Remove a terminal job record and clear the current selection. */
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
/** Clear terminal job history and synchronize shared runtime state. */
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
/** Delete local artifacts for the selected job after confirmation. */
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
