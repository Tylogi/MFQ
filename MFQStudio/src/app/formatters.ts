/** 统一应用的错误、数值、容量和时长展示格式。 */
import { ApiError } from '../shared/api/client';

/** 将服务端或普通异常转换为用户可见的错误信息。 */
export function errorMessage(error: unknown, tr?: (chinese: string, english: string) => string): string {
  if (error instanceof ApiError) {
    if (tr) {
      const port = Number.isInteger(error.details?.port) ? ` ${error.details.port}` : '';
      const messages: Record<string, [string, string]> = {
        listener_port_conflict: ['OpenAI 和 Anthropic 不能使用同一个端口，请为它们设置不同的端口。', 'OpenAI and Anthropic cannot share a port. Choose a different port for each API.'],
        listener_port_in_use: [`端口${port}已被其它服务占用，请选择另一个端口。`, `Port${port} is already in use. Choose another port.`],
        listener_permission_denied: [`没有权限使用端口${port}，请选择其它端口。`, `Permission denied for port${port}. Choose another port.`],
        listener_change_failed: ['端口设置未能保存，原端口保持不变。请检查配置目录是否可写后重试。', 'The port change could not be saved. The previous port is unchanged. Check that the configuration folder is writable and retry.'],
        listener_not_managed: ['当前端口由外部程序管理，请在启动该服务的程序中修改。', 'This listener is managed externally. Change its port in the program that starts the service.'],
        listener_not_ready: ['服务尚未就绪，请稍后再修改端口。', 'The service is not ready. Try changing the port again shortly.'],
        invalid_authentication_policy: ['API Key 必须为 1–512 个可打印英文字符且不含空格；启用鉴权时不能为空。', 'API key must contain 1–512 printable ASCII characters without spaces and cannot be empty when enabling authentication.'],
        authentication_required: ['对外监听的服务必须启用鉴权，不能设置为无。', 'Authentication is required for a network listener and cannot be disabled.'],
        authentication_save_failed: ['鉴权设置未能保存，原设置保持不变。请检查配置目录是否可写。', 'Authentication settings could not be saved. The previous settings are unchanged. Check that the configuration folder is writable.'],
        invalid_download_directory: ['下载目录不可用，请选择一个存在且可写的文件夹。', 'The download folder is unavailable. Choose an existing writable folder.'],
        download_directory_save_failed: ['下载目录未能保存，原目录保持不变。请检查配置目录是否可写。', 'The download folder could not be saved. The previous folder is unchanged. Check that the configuration folder is writable.'],
        insufficient_disk_space: ['可用磁盘空间不足，或已被其它下载预留。可勾选“驱逐 SSD Prefix 预算”回收前缀缓存，或更换下载目录。', 'Available disk space is insufficient or reserved by other downloads. Enable “Reclaim SSD prefix budget” or choose another download folder.'],
        download_size_unknown: ['尚未获取完整文件大小，请刷新模型文件列表后再下载。', 'File sizes are not available yet. Refresh the model file list before downloading.'],
        download_destination_busy: ['此目录已有下载任务，请等待完成或先取消该任务。', 'This folder already has an active download. Wait for it to finish or cancel it first.'],
        download_already_exists: ['目标目录已存在完整文件，无需重复下载。', 'The complete files already exist in the destination folder. No download is needed.'],
        cache_reclaim_busy: ['前缀缓存正在使用或模型正在加载，请稍后重试下载。', 'The prefix cache is in use or a model is loading. Retry the download shortly.'],
        cache_reclaim_failed: ['未能释放足够的前缀缓存空间，请稍后重试或更换下载目录。', 'Not enough prefix cache space could be released. Retry or choose another download folder.'],
        model_directory_not_found: ['找不到此文件夹，请检查路径或重新选择。', 'This folder could not be found. Check the path or choose it again.'],
        directory_not_found: ['找不到此文件夹，请检查路径或重新选择。', 'This folder could not be found. Check the path or choose it again.'],
        not_a_directory: ['此位置是文件而不是文件夹，请选择一个目录。', 'This location is a file, not a folder. Choose a directory.'],
        quantization_validation_failed: ['模型或文件不符合量化要求，请检查路径、格式和文件是否完整。', 'The model or file does not meet the quantization requirements. Check the path, format and file completeness.'],
        invalid_output_extension: ['输出文件扩展名不正确，请使用此任务要求的文件格式。', 'The output file extension is incorrect. Use the format required by this task.'],
        output_exists: ['输出文件已经存在，请选择其它文件名，避免覆盖已有文件。', 'The output file already exists. Choose another filename to avoid overwriting it.'],
        model_registration_failed: ['此文件夹中没有可登记的模型，请选择包含完整模型文件的目录。', 'This folder has no model that can be registered. Choose a folder containing complete model files.'],
        invalid_model_alias: ['模型别名不能包含控制字符，长度不能超过 255 个字符。', 'Model aliases cannot contain control characters or exceed 255 characters.'],
        model_alias_conflict: ['模型别名不能重复，也不能与其它模型的名称相同。', 'Model aliases must be unique and cannot match another model’s name.'],
        cache_not_found: ['此缓存已被清理或淘汰，请刷新列表。', 'This cache was cleared or evicted. Refresh the list.'],
        resident_budget_exhausted: ['预算不足以容纳不可卸载的权重和活动 KV，请提高预算或卸载其它模型。', 'The budget cannot accommodate non-offloadable weights and active KV. Increase it or unload another model.'],
        kv_memory_insufficient: ['可用显存和内存不足，请选择自动（启用SSD KV），这可能会拖慢长上下文时的速度并造成磁盘占用，或者降低上下文长度。', 'Available VRAM and RAM are insufficient. Select Automatic (with SSD KV), which may slow long-context inference and use disk space, or reduce the context length.'],
        kv_vram_insufficient: ['可用显存不足，请选择自动档或降低上下文长度。', 'Available VRAM is insufficient. Select automatic placement or reduce the context length.'],
        kv_dynamic_unavailable: ['当前配置不支持实验性动态 KV，请使用预分配，或选择全显存 / 自动（不启用 SSD KV）。', 'Experimental dynamic KV is unavailable for this configuration. Use preallocation, VRAM only, or automatic without SSD.'],
        kv_memory_unsupported: ['当前模型后端尚不支持 KV 存储位置设置。', 'This model backend does not support configurable KV placement.'],
        resident_budget_unsupported: ['当前后端不支持在线调整此预算，模型未被重载。请更新后端后重试。', 'This backend cannot adjust this budget online. Models were not reloaded. Update the backend and retry.'],
        yarn_unsupported: ['此模型或后端不支持 YaRN，不能启用上下文扩展。', 'This model or backend does not support YaRN context extension.'],
        qsa_kv_configuration_invalid: ['流式稀疏注意力设置不符合要求，请检查预算和最大上下文。', 'The streaming sparse attention settings are invalid. Check the budget and maximum context.'],
        qsa_kv_offload_unsupported: ['当前模型不支持流式稀疏注意力，请关闭该选项。', 'This model does not support streaming sparse attention. Disable this option.'],
        runtime_busy: ['运行时正在处理请求或设置，请稍后重试。', 'The runtime is handling requests or settings. Try again shortly.'],
        model_not_loaded: ['模型尚未加载，请先加载模型。', 'The model is not loaded. Load it first.'],
        model_delete_busy: ['模型正在加载或使用中，请先卸载再删除。', 'The model is loading or in use. Unload it before deleting.'],
        model_delete_downloading: ['此模型仍在下载，请等待完成或先取消下载。', 'This model is still downloading. Wait for it to finish or cancel the download first.'],
        model_delete_changed: ['模型文件已发生变化，请关闭确认框并刷新列表后重试。', 'The model files changed. Close the dialog, refresh the list and try again.'],
        model_delete_unsafe: ['无法安全确定此模型的文件范围，未执行删除。', 'The model files could not be safely identified. Nothing was deleted.'],
        model_delete_failed: ['部分文件未能删除，请检查文件权限并刷新列表。', 'Some files could not be deleted. Check file permissions and refresh the list.'],
        model_artifact_not_found: ['此模型已不存在，请刷新列表。', 'This model no longer exists. Refresh the list.'],
        model_delete_unavailable: ['当前服务不支持安全删除模型，请更新后端后重试。', 'This server does not support safe model deletion. Update the backend and retry.'],
        model_not_ready: ['模型尚未就绪，请等待加载完成。', 'The model is not ready. Wait for loading to finish.'],
        invalid_request: ['输入值不符合要求，请检查数值范围和格式。', 'Some input values are invalid. Check their ranges and format.'],
      };
      if (error.code === 'listener_change_failed' && /address already in use|EADDRINUSE/i.test(error.message)) return tr(...messages.listener_port_in_use);
      if (error.code === 'invalid_request' && Array.isArray(error.details?.errors)) {
        const fields = error.details.errors.flatMap(value => value && typeof value === 'object' && Array.isArray(value.loc) ? value.loc : []);
        if (fields.includes('port')) return tr('端口必须为 1–65535 的整数。', 'Port must be an integer between 1 and 65535.');
        if (fields.includes('context_size') || fields.includes('max_context_size')) return tr('请输入不小于 512 的有效上下文整数，或留空使用自动设置。', 'Enter a valid context integer of at least 512, or leave empty for automatic sizing.');
      }
      if (messages[error.code]) return tr(...messages[error.code]);
    }
    return `${error.code}: ${error.message}`;
  }
  if (error instanceof Error) return error.message;
  if (typeof error === 'string' && error.trim()) return error;
  if (error && typeof error === 'object' && 'message' in error && typeof error.message === 'string' && error.message.trim()) return error.message;
  return "Unknown error";
}

const numberFormatters = new Map<number, Intl.NumberFormat>();

function numberFormatter(digits: number): Intl.NumberFormat {
  const key = Math.max(0, Math.min(20, Math.trunc(digits)));
  let formatter = numberFormatters.get(key);
  if (!formatter) {
    formatter = new Intl.NumberFormat(undefined, { maximumFractionDigits: key });
    numberFormatters.set(key, formatter);
  }
  return formatter;
}

/** 格式化有限数值，缺失或无效时返回占位符。 */
export function formatNumber(value: unknown, digits = 0): string {
  const number = Number(value);
  return Number.isFinite(number) ? numberFormatter(digits).format(number) : "--";
}

/** 将字节数转换为可读的二进制容量单位。 */
export function formatBytes(value: unknown): string {
  const bytes = Number(value);
  if (!Number.isFinite(bytes) || bytes <= 0) return "--";
  if (bytes >= 2 ** 40) return `${formatNumber(bytes / 2 ** 40, 1)} TB`;
  if (bytes >= 2 ** 30) return `${formatNumber(bytes / 2 ** 30, 1)} GB`;
  if (bytes >= 2 ** 20) return `${formatNumber(bytes / 2 ** 20, 0)} MB`;
  return `${formatNumber(bytes / 2 ** 10, 0)} KB`;
}

/** 将秒数转换为天、小时、分钟与秒的简洁时长。 */
export function formatDuration(value: unknown): string {
  const seconds = Math.max(0, Math.floor(Number(value) || 0));
  const days = Math.floor(seconds / 86400);
  const hours = Math.floor((seconds % 86400) / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  if (days) return `${days}d ${hours}h`;
  if (hours) return `${hours}h ${minutes}m`;
  return `${minutes}m ${seconds % 60}s`;
}
