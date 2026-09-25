/** 解析任务表单字段的类型与初始值，并识别任务终止状态。 */
import { JobResource, JsonSchemaProperty } from '../../shared/api/types';

/** 根据任务字段定义生成初始值，支持可空联合类型。 */
export function schemaDefault(property: JsonSchemaProperty): unknown {
  if (property.default !== undefined) return property.default;
  const option = property.anyOf?.find((item) => item.type && item.type !== "null");
  if (option) return schemaDefault(option);
  if (property.type === "boolean") return false;
  if (property.type === "array") return [];
  return "";
}

/** 解析任务字段的非空类型，供表单选择输入控件。 */
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

/** 判断后台任务是否已经结束，供轮询停止与状态展示使用。 */
export function isTerminalJob(job: JobResource): boolean {
  return TERMINAL_JOB_STATUSES.has(job.status);
}
