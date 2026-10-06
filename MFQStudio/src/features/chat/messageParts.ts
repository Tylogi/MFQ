/** Extract message text, reasoning, and displayable media parts from chat messages. */
import { ContentPart, Message } from '../../shared/api/types';
/** Combine message text and reasoning parts for saved-message display and editing. */
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
/** Narrow a message part to a playable or displayable media type. */
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
