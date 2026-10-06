/** Load and display chat media and documents, managing download state, video posters, and resource cleanup. */
import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { mediaApi } from '../../shared/api/resources/media';
import type { ContentPart } from '../../shared/api/types';
import { formatNumber } from '../../app/formatters';
/** Display video and extract its first-frame poster, releasing the decoder and object URL on unmount. */
export function VideoWithFirstFrame({
  className,
  controls = false,
  muted = false,
  src,
}: {
  className?: string;
  controls?: boolean;
  muted?: boolean;
  src: string;
}) {
  const [poster, setPoster] = useState<string | null>(null);

  useEffect(() => {
    const video = document.createElement("video");
    let cancelled = false;
    let posterUrl: string | null = null;
    setPoster(null);
    video.muted = true;
    video.playsInline = true;
    video.preload = "auto";
    video.onloadeddata = () => {
      video.onloadeddata = null;
      if (cancelled || !video.videoWidth || !video.videoHeight) return;
      const scale = Math.min(1, 1280 / video.videoWidth);
      const canvas = document.createElement("canvas");
      canvas.width = Math.max(1, Math.round(video.videoWidth * scale));
      canvas.height = Math.max(1, Math.round(video.videoHeight * scale));
      canvas.getContext("2d")?.drawImage(video, 0, 0, canvas.width, canvas.height);
      canvas.toBlob((blob) => {
        if (cancelled || !blob) return;
        posterUrl = URL.createObjectURL(blob);
        setPoster(posterUrl);
      }, "image/jpeg", 0.85);
    };
    video.src = src;
    video.load();
    return () => {
      cancelled = true;
      video.onloadeddata = null;
      video.pause();
      video.removeAttribute("src");
      video.load();
      if (posterUrl) URL.revokeObjectURL(posterUrl);
    };
  }, [src]);

  return <video className={className} controls={controls} muted={muted} playsInline poster={poster ?? undefined} preload="metadata" src={src} />;
}
/** Load an attachment by message-media ID, supporting request cancellation and failure state. */
export function MediaPartView({ part }: { part: Extract<ContentPart, { media: unknown }> }) {
  const { t } = useTranslation();
  const [src, setSrc] = useState<string | null>(null);
  const [loadFailed, setLoadFailed] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    let objectUrl: string | null = null;
    setSrc(null);
    setLoadFailed(false);
    void mediaApi.fetchMedia(part.media.id, controller.signal).then((blob) => {
      if (controller.signal.aborted) return;
      objectUrl = URL.createObjectURL(blob);
      setSrc(objectUrl);
    }).catch((error: unknown) => {
      if (!controller.signal.aborted) {
        console.error("Unable to load message media", error);
        setLoadFailed(true);
      }
    });
    return () => {
      controller.abort();
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [part.media.id]);

  if (loadFailed) {
    return <span className="message-media-status" role="alert">{t('chat:media.unableToLoadAttachment')}</span>;
  }
  if (!src) return <span className="message-media-status" aria-label={t('chat:media.loadingMedia')}>{t('chat:media.loadingAttachment')}</span>;
  if (part.type === "image") {
    return <img alt={t('chat:media.attachedImage')} className="message-media media-image" loading="lazy" src={src} />;
  }
  if (part.type === "video") {
    return <VideoWithFirstFrame className="message-media media-video" controls src={src} />;
  }
  return <audio className="message-audio" controls preload="metadata" src={src} />;
}
/** Display document attachments with download, progress, and retry interactions. */
export function DocumentPartView({ part }: { part: Extract<ContentPart, { type: "document" }> }) {
  const { t } = useTranslation();
  const [downloadState, setDownloadState] = useState<"idle" | "loading" | "failed">("idle");

  async function downloadDocument() {
    if (downloadState === "loading") return;
    setDownloadState("loading");
    try {
      const blob = await mediaApi.fetchMedia(part.media.id);
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = part.name;
      anchor.hidden = true;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
      setDownloadState("idle");
    } catch (error) {
      console.error("Unable to download document", error);
      setDownloadState("failed");
    }
  }

  const detail = downloadState === "loading"
    ? t('chat:media.downloading')
    : downloadState === "failed"
      ? t('chat:media.downloadFailed')
      : `${formatNumber(part.media.byte_size)} B`;
  return <button className="message-document" disabled={downloadState === "loading"} onClick={() => void downloadDocument()} type="button"><span>DOC</span><div><strong>{part.name}</strong><small>{detail}</small></div></button>;
}
