/** 汇总运行时模型、实例与加载任务，保持模型选择状态一致。 */
import { JobResource, RuntimeInstance, RuntimeModel } from '../../shared/api/types';

/** 判断实例是否已加载并可承接推理请求。 */
export function isRuntimeReady(state: string | null | undefined): boolean {
  return state === "ready" || state === "busy";
}

/** 合并服务公布和已就绪实例的模型名称并去重。 */
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

/** 合并就绪、加载中实例与加载任务，生成可选模型名称。 */
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
