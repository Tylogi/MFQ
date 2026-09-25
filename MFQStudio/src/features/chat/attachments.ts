/** 管理聊天附件类型、大小限制、文档识别及媒体元数据读取。 */


export interface PendingAttachment {
  id: string;
  file: File;
  previewUrl: string;
  kind: "image" | "video" | "audio" | "document";
}

export const DOCUMENT_ACCEPT = [
  ".txt",
  ".md",
  ".markdown",
  ".json",
  ".jsonl",
  ".csv",
  ".tsv",
  ".yaml",
  ".yml",
  ".xml",
  ".html",
  ".css",
  ".py",
  ".js",
  ".jsx",
  ".ts",
  ".tsx",
  ".c",
  ".cc",
  ".cpp",
  ".h",
  ".hpp",
  ".rs",
  ".go",
  ".java",
  ".sh",
  ".toml",
  ".ini",
  ".log",
  ".pdf",
  ".docx",
].join(",");

export const MAX_DOCUMENT_BYTES = 64 * 1024 * 1024;

/** 根据 MIME 和扩展名判断附件是否可作为文本读取。 */
export function isTextDocument(file: File): boolean {
  const extension = file.name.toLowerCase().match(/\.[^.]+$/)?.[0] ?? "";
  return (
    file.type.startsWith("text/") ||
    ["application/json", "application/xml", "application/yaml"].includes(file.type) ||
    DOCUMENT_ACCEPT.split(",").includes(extension)
  );
}

/** 读取媒体尺寸或音频参数，并在完成后释放临时资源。 */
export async function mediaMetadata(
  file: File,
  kind: Exclude<PendingAttachment["kind"], "document">,
) {
  if (kind === "image") {
    const bitmap = await createImageBitmap(file);
    const result = { width: bitmap.width, height: bitmap.height };
    bitmap.close();
    return result;
  }
  if (kind === "audio") {
    const context = new AudioContext();
    try {
      const buffer = await context.decodeAudioData(await file.arrayBuffer());
      return {
        sample_rate_hz: buffer.sampleRate,
        channels: buffer.numberOfChannels,
        duration_ms: Math.round(buffer.duration * 1000),
      };
    } finally {
      await context.close();
    }
  }
  const url = URL.createObjectURL(file);
  try {
    return await new Promise<{ width: number; height: number; duration_ms: number }>(
      (resolve, reject) => {
        const video = document.createElement("video");
        const finish = (
          callback: () => void,
        ) => {
          window.clearTimeout(timeout);
          video.onloadedmetadata = null;
          video.onerror = null;
          video.removeAttribute("src");
          video.load();
          callback();
        };
        const timeout = window.setTimeout(
          () => finish(() => reject(new Error("Unable to read video metadata"))),
          15_000,
        );
        video.preload = "metadata";
        video.onloadedmetadata = () => {
          const width = video.videoWidth;
          const height = video.videoHeight;
          const durationMs = Math.round(video.duration * 1000);
          if (!width || !height || !Number.isFinite(durationMs)) {
            finish(() => reject(new Error("Unable to read video metadata")));
            return;
          }
          finish(() => resolve({ width, height, duration_ms: durationMs }));
        };
        video.onerror = () => finish(() => reject(new Error("Unable to read video metadata")));
        video.src = url;
        video.load();
      },
    );
  } finally {
    URL.revokeObjectURL(url);
  }
}

/** 优先使用文件 MIME，并为常见文档扩展名提供默认值。 */
export function documentMimeType(file: File): string {
  if (file.type) return file.type;
  const extension = file.name.toLowerCase().match(/\.[^.]+$/)?.[0] ?? "";
  if (extension === ".pdf") return "application/pdf";
  if (extension === ".docx") {
    return "application/vnd.openxmlformats-officedocument.wordprocessingml.document";
  }
  if (extension === ".json" || extension === ".jsonl") return "application/json";
  if (extension === ".xml") return "application/xml";
  return "text/plain";
}
