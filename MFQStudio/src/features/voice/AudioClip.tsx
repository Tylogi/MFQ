/** Load recordings from local voice history and manage the player's object URL lifecycle. */
import { useEffect, useState } from 'react';
import { loadVoiceClip } from '../../realtimeAudio';

/** Load and play a recording from local voice storage, releasing its object URL on unmount. */
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
