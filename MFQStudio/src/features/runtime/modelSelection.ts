/** Aggregate runtime models, instances, and loading jobs to keep model selection consistent. */
import { JobResource, RuntimeInstance, RuntimeModel } from '../../shared/api/types';

/** Determine whether an instance is loaded and ready to handle inference requests. */
export function isRuntimeReady(state: string | null | undefined): boolean {
  return state === "ready" || state === "busy";
}

/** Merge and deduplicate model names published by the server and ready instances. */
export function runtimeModelNames(
  advertised: RuntimeModel[],
  instances: RuntimeInstance[],
): string[] {
  return Array.from(new Set([
    ...advertised.map((item) => item.id),
    ...instances
      .filter((item) => isRuntimeReady(item.state))
      .map((item) => item.model),
  ].filter(Boolean)));
}

/** Merge ready and loading instances with loading jobs to produce selectable model names. */
export function runtimeSelectionNames(
  advertised: RuntimeModel[],
  instances: RuntimeInstance[],
  jobs: JobResource[] = [],
): string[] {
  return Array.from(new Set([
    ...runtimeModelNames(advertised, instances),
    ...instances
      .filter((item) => item.state !== "failed" && item.state !== "unloading")
      .map((item) => item.model),
    ...jobs
      .filter((item) => item.kind === "model.load"
        && ["queued", "running", "cancelling"].includes(item.status))
      .map((item) => item.payload.model)
      .filter((item): item is string => typeof item === "string" && Boolean(item)),
  ]));
}
