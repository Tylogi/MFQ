/** 加载和展示聊天媒体与文档，管理下载状态、视频首帧和资源释放。 */
import { useEffect, useState } from 'react';
import { mediaApi } from '../../shared/api/resources/media';
import type { ContentPart } from '../../shared/api/types';
import { formatNumber } from '../../app/formatters';

/** 展示视频并提取首帧海报，卸载时释放解码器和对象 URL。 */
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

/** 按消息媒体标识加载附件，支持取消请求及失败状态。 */
export function MediaPartView({ part }: { part: Extract<ContentPart, { media: unknown }> }) {
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
    return <span className="message-media-status" role="alert">Unable to load attachment</span>;
  }
  if (!src) return <span className="message-media-status" aria-label="Loading media">Loading attachment…</span>;
  if (part.type === "image") {
    return <img alt="Attached image" className="message-media media-image" loading="lazy" src={src} />;
  }
  if (part.type === "video") {
    return <VideoWithFirstFrame className="message-media media-video" controls src={src} />;
  }
  return <audio className="message-audio" controls preload="metadata" src={src} />;
}

/** 展示文档附件并提供下载、进度及失败重试交互。 */
export function DocumentPartView({ part }: { part: Extract<ContentPart, { type: "document" }> }) {
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
    ? "Downloading…"
    : downloadState === "failed"
      ? "Download failed — click to retry"
      : `${formatNumber(part.media.byte_size)} B`;
  return <button className="message-document" disabled={downloadState === "loading"} onClick={() => void downloadDocument()} type="button"><span>DOC</span><div><strong>{part.name}</strong><small>{detail}</small></div></button>;
}
