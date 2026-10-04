/** Load modes and defaults and submit the dynamic job-parameter form. */
import { useEffect, useState, type FormEvent, type Dispatch, type SetStateAction } from 'react';
import type { JobKindResource, JsonSchemaProperty } from '../../shared/api/types';
import { jobsApi } from '../../shared/api/resources/jobs';
import { schemaDefault, schemaType } from './jobSchema';
import { errorMessage } from '../../app/formatters';
import { toast } from '../../stores/toastStore';
import { useJobStore } from '../../stores/jobStore';
/** Maintain the job type and payload, selecting the created job after successful submission. */
export function useJobForm(setSelectedJobId: Dispatch<SetStateAction<string | null>>) {
  const addJob = useJobStore((state) => state.addJob);
  const [busy, setBusy] = useState(false);
  const [jobKinds, setJobKinds] = useState<JobKindResource[]>([]);
  const [selectedJobKind, setSelectedJobKind] = useState('');
  const [jobPayload, setJobPayload] = useState<Record<string, unknown>>({});
  const [pendingImatrix, setPendingImatrix] = useState('');
  useEffect(() => {
    let active = true;
    void jobsApi.jobKinds().then((kinds) => {
      if (!active) return;
      setJobKinds(kinds);
      setSelectedJobKind((kind) => kind || kinds[0]?.kind || '');
    }).catch((cause) => {
      if (active) toast.error(errorMessage(cause));
    });
    return () => { active = false; };
  }, []);
  useEffect(() => {
    const properties =
      jobKinds.find((item) => item.kind === selectedJobKind)?.payload_schema.properties ?? {};
    setJobPayload(Object.fromEntries(Object.entries(properties).map(([name, property]) => [
      name,
      selectedJobKind === 'model.quantize' && name === 'imatrix' && pendingImatrix
        ? pendingImatrix
        : schemaDefault(property),
    ])));
  }, [jobKinds, selectedJobKind, pendingImatrix]);
/** Convert form values according to the job-parameter mode. */
  function updateJobPayload(name: string, property: JsonSchemaProperty, value: string | boolean) {
    const type = schemaType(property);
    let parsed: unknown = value;
    if (type === 'integer' || type === 'number') {
      parsed = value === '' ? null : Number(value);
    } else if (type === 'array') {
      parsed = String(value).split(',').map((item) => item.trim()).filter(Boolean)
        .map((item) => (property.items?.type === 'integer' ? Number(item) : item));
    }
    setJobPayload((current) => ({ ...current, [name]: parsed }));
  }
/** Submit the current parameters and select the new job. */
  async function submitJob(event: FormEvent) {
    event.preventDefault();
    if (!selectedJobKind) return;
    setBusy(true);
    try {
      const clean = Object.fromEntries(
        Object.entries(jobPayload).filter(([, value]) => value !== '' && value !== null),
      );
      const created = await jobsApi.createJob(selectedJobKind, clean);
      setSelectedJobId(created.id);
      addJob(created);
    } catch (cause) {
      toast.error(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  return {
    busy, setBusy, jobKinds, selectedJobKind, setSelectedJobKind, jobPayload,
    setPendingImatrix, updateJobPayload, submitJob,
  };
}
