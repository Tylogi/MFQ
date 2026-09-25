/** 量化页面组合任务表单、产物、历史操作与日志订阅。 */
import { useEffect, useState } from 'react';
import { useLocation } from 'react-router';
import { useSettings } from '../settings/SettingsProvider';
import { useJobStore } from '../../stores/jobStore';
import { isTerminalJob } from './jobSchema';
import { useJobForm } from './useJobForm';
import { useJobEventLog } from './useJobEventLog';
import { useJobArtifacts } from './useJobArtifacts';
import { useJobRecordActions } from './useJobRecordActions';

/** 按需组装量化资源，后台任务生命周期由共享运行时维持。 */
export function useQuantizationWorkspace() {
  const { tr } = useSettings();
  const jobs = useJobStore((state) => state.jobs);
  const location = useLocation();
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const selectedJob = jobs.find((item) => item.id === selectedJobId) ?? null;
  const activeJobs = jobs.filter((item) => !isTerminalJob(item));
  const completedJobs = jobs.filter(isTerminalJob);
  const form = useJobForm(setSelectedJobId);
  const logs = useJobEventLog(selectedJobId);
  const completedVersion = completedJobs.map((job) => job.id + ':' + job.updated_at).join(',');
  const artifacts = useJobArtifacts(completedVersion, form.busy, setSelectedJobId);
  const actions = useJobRecordActions(
    selectedJobId, setSelectedJobId, selectedJob, form.busy, form.setBusy, logs.setJobLogs,
  );
  useEffect(() => {
    const jobId = (location.state as { jobId?: string } | null)?.jobId;
    if (jobId) setSelectedJobId(jobId);
  }, [location.state]);

  return {
    tr,
    busy: form.busy,
    error: null,
    panelLabels: {
      collapse: tr('折叠面板', 'Collapse panel'),
      expand: tr('展开面板', 'Expand panel'),
    },
    lineage: artifacts.lineage,
    jobKinds: form.jobKinds,
    selectedJobKind: form.selectedJobKind,
    setSelectedJobKind: form.setSelectedJobKind,
    jobPayload: form.jobPayload,
    selectedJobId,
    setSelectedJobId,
    imatrixImporting: artifacts.imatrixImporting,
    setPendingImatrix: form.setPendingImatrix,
    imatrixInputRef: artifacts.imatrixInputRef,
    jobLogs: logs.jobLogs,
    jobCleanupBusy: actions.jobCleanupBusy,
    selectedJob,
    activeJobs,
    completedJobs,
    genericJobKinds: form.jobKinds,
    selectedKind: form.jobKinds.find((item) => item.kind === form.selectedJobKind),
    imatrixArtifacts: artifacts.imatrixArtifacts,
    updateJobPayload: form.updateJobPayload,
    submitJob: form.submitJob,
    importImatrix: artifacts.importImatrix,
    cancelSelectedJob: actions.cancelSelectedJob,
    retrySelectedJob: actions.retrySelectedJob,
    deleteJobRecord: actions.deleteJobRecord,
    clearCompletedJobRecords: actions.clearCompletedJobRecords,
    removeSelectedArtifact: actions.removeSelectedArtifact,
  };
}
