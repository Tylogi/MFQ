/** 量化任务关联的产物谱系与 Imatrix 导入流程。 */
import { useEffect, useRef, useState, type Dispatch, type SetStateAction } from 'react';
import type { ArtifactLineage } from '../../shared/api/types';
import { modelsApi } from '../../shared/api/resources/models';
import { jobsApi } from '../../shared/api/resources/jobs';
import { mediaApi } from '../../shared/api/resources/media';
import { errorMessage } from '../../app/formatters';
import { toast } from '../../stores/toastStore';
import { useJobStore } from '../../stores/jobStore';

/** 随终态任务版本刷新谱系，并为上传文件创建 Imatrix 导入任务。 */
export function useJobArtifacts(
  completedVersion: string,
  busy: boolean,
  setSelectedJobId: Dispatch<SetStateAction<string | null>>,
) {
  const addJob = useJobStore((state) => state.addJob);
  const [lineage, setLineage] = useState<ArtifactLineage[]>([]);
  const [imatrixImporting, setImatrixImporting] = useState(false);
  const imatrixInputRef = useRef<HTMLInputElement | null>(null);
  useEffect(() => {
    let active = true;
    void modelsApi.artifactLineage().then((items) => {
      if (active) setLineage(items);
    }).catch((cause) => {
      if (active) toast.error(errorMessage(cause));
    });
    return () => { active = false; };
  }, [completedVersion]);
  const imatrixArtifacts = lineage.filter((item) =>
    item.producer_kind === 'calibrate.imatrix' ||
    item.producer_kind === 'artifact.import' ||
    item.metadata?.media_type === 'application/x-mfq-imatrix' ||
    item.artifact_name.endsWith('.imatrix'),
  );

  /** 上传并创建 Imatrix 导入任务。 */
  async function importImatrix(files: FileList | null) {
    const file = files?.[0];
    if (!file || imatrixImporting || busy) return;
    setImatrixImporting(true);
    try {
      const uploaded = await mediaApi.uploadMedia(file, 'application/x-mfq-imatrix');
      const destination = `artifacts/imatrix/${file.name.replace(/[^A-Za-z0-9_.-]+/g, '-')}`;
      const created = await jobsApi.createJob('artifact.import', {
        media_id: uploaded.media.id, destination, kind: 'imatrix',
      });
      setSelectedJobId(created.id);
      addJob(created);
    } catch (cause) {
      toast.error(errorMessage(cause));
    } finally {
      setImatrixImporting(false);
      if (imatrixInputRef.current) imatrixInputRef.current.value = '';
    }
  }
  return { lineage, imatrixArtifacts, imatrixImporting, imatrixInputRef, importImatrix };
}
