/** 从聊天消息中提取正文、推理文本与可展示的媒体片段。 */
import { ContentPart, Message } from '../../shared/api/types';

/** 合并消息中的正文和思考片段，供历史消息展示与编辑。 */
export function textParts(message: Message): { text: string; reasoning: string } {
  const text = message.parts
    .filter((part) => part.type === "text" || part.type === "transcript")
    .map((part) => part.text)
    .join("");
  const reasoning = message.parts
    .filter((part) => part.type === "reasoning")
    .map((part) => part.text)
    .join("");
  return { text, reasoning };
}

/** 收窄消息片段为可播放或展示的媒体类型。 */
export function isMediaPart(
  part: ContentPart,
): part is Extract<ContentPart, { type: "image" | "video" | "audio" | "generated_audio" }> {
  return (
    part.type === "image" ||
    part.type === "video" ||
    part.type === "audio" ||
    part.type === "generated_audio"
  );
}
