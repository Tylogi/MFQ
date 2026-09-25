/** 选中任务的历史事件与实时事件订阅。 */
import { useEffect, useState } from 'react';
import type { RuntimeLogEntry } from '../../shared/api/types';
import { jobsApi } from '../../shared/api/resources/jobs';
import { errorMessage } from '../../app/formatters';
import { toast } from '../../stores/toastStore';

/** 订阅当前任务日志，切换任务时取消旧订阅并清空旧内容。 */
export function useJobEventLog(selectedJobId: string | null) {
  const [jobLogs, setJobLogs] = useState<RuntimeLogEntry[]>([]);
  useEffect(() => {
    setJobLogs([]);
    if (!selectedJobId) return;
    const controller = new AbortController();
    void jobsApi.jobEvents(selectedJobId)
      .then((entries) => {
        if (controller.signal.aborted) return;
        setJobLogs((current) =>
          [...new Map([...entries, ...current].map((entry) => [entry.sequence, entry])).values()]
            .sort((left, right) => left.sequence - right.sequence),
        );
      })
      .catch((cause) => {
        if (!controller.signal.aborted) toast.error(errorMessage(cause));
      });
    void jobsApi.streamJobEvents(
      selectedJobId,
      (event) => {
        if (controller.signal.aborted || !event.message) return;
        setJobLogs((current) =>
          current.some((entry) => entry.sequence === event.sequence)
            ? current
            : [...current, {
                sequence: event.sequence,
                level: event.level,
                message: event.message || '',
                fields: event.data,
                created_at: event.created_at,
              }].sort((left, right) => left.sequence - right.sequence),
        );
      },
      controller.signal,
    ).catch((cause) => {
      if (!controller.signal.aborted) toast.error(errorMessage(cause));
    });
    return () => controller.abort();
  }, [selectedJobId]);
  return { jobLogs, setJobLogs };
}
