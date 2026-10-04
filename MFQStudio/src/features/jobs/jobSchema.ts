/** Parse job form field types and initial values, and identify terminal job states. */
import { JobResource, JsonSchemaProperty } from '../../shared/api/types';
/** Generate initial values from job field definitions, supporting nullable union types. */
export function schemaDefault(property: JsonSchemaProperty): unknown {
  if (property.default !== undefined) return property.default;
  const option = property.anyOf?.find((item) => item.type && item.type !== "null");
  if (option) return schemaDefault(option);
  if (property.type === "boolean") return false;
  if (property.type === "array") return [];
  return "";
}
/** Resolve a job field’s non-null type for form selection controls. */
export function schemaType(property: JsonSchemaProperty): string | undefined {
  if (typeof property.type === "string") return property.type;
  return property.anyOf?.find((item) => item.type && item.type !== "null")?.type as
    | string
    | undefined;
}

export const TERMINAL_JOB_STATUSES = new Set<JobResource["status"]>([
  "succeeded",
  "failed",
  "cancelled",
  "interrupted",
]);
/** Determine whether a background job has finished for polling and status display. */
export function isTerminalJob(job: JobResource): boolean {
  return TERMINAL_JOB_STATUSES.has(job.status);
}
