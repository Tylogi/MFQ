/** 加载本地语音记录中的录音，管理播放器对象 URL 的生命周期。 */
import { useEffect, useState } from 'react';
import { loadVoiceClip } from '../../realtimeAudio';

/** 从本地语音存储加载录音并播放，卸载后释放对象 URL。 */
export function AudioClip({ audioId }: { audioId: string }) {
  const [url, setUrl] = useState("");
  useEffect(() => {
    let current = true;
    let objectUrl = "";
    loadVoiceClip(audioId)
      .then((blob) => {
        if (!blob || !current) return;
        objectUrl = URL.createObjectURL(blob);
        setUrl(objectUrl);
      })
      .catch(() => undefined);
    return () => {
      current = false;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [audioId]);
  return url ? <audio className="message-audio" controls preload="metadata" src={url} /> : null;
}
