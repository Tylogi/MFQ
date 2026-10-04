/** Verify authentication, first frames, metadata validation, and resource cleanup using real components, media APIs, and attachment hooks; browser decoding APIs are driven by event substitutes. */
import { act, cleanup, fireEvent, render, renderHook, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { DocumentPartView, MediaPartView, VideoWithFirstFrame } from '../src/features/chat/MessageMedia';
import { mediaMetadata } from '../src/features/chat/attachments';
import { useChatAttachments } from '../src/features/chat/hooks/useChatAttachments';
import { mediaApi } from '../src/shared/api/resources/media';
import { setApiBaseUrl, setApiToken } from '../src/shared/api/client';

const media = { id: 'protected-media', sha256: 'digest', mime_type: 'image/png', byte_size: 8 };
const createObjectURL = vi.fn<(blob: Blob) => string>();
const revokeObjectURL = vi.fn<(url: string) => void>();

beforeEach(() => {
  createObjectURL.mockReset().mockReturnValue('blob:protected');
  revokeObjectURL.mockReset();
  vi.stubGlobal('URL', class extends URL {
    static createObjectURL = createObjectURL;
    static revokeObjectURL = revokeObjectURL;
  });
  vi.spyOn(HTMLMediaElement.prototype, 'load').mockImplementation(() => {});
  vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {});
  setApiBaseUrl('https://studio.example');
  setApiToken('private-token');
});

afterEach(() => {
  cleanup();
  setApiBaseUrl('');
  setApiToken('');
  vi.useRealTimers();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

/** Intercept offscreen video creation while retaining real DOM methods for event and cleanup assertions. */
function captureDecoder() {
  const createElement = document.createElement.bind(document);
  const video = createElement('video');
  vi.spyOn(document, 'createElement').mockImplementation((tag, options) =>
    tag === 'video' ? video : createElement(tag, options),
  );
  return video;
}

/** Simulate video dimensions and duration read by the browser without replacing the metadata logic under test. */
function setVideoMetadata(video: HTMLVideoElement, width = 1920, height = 1080, duration = 1.25) {
  Object.defineProperties(video, {
    videoWidth: { configurable: true, value: width },
    videoHeight: { configurable: true, value: height },
    duration: { configurable: true, value: duration },
  });
}

describe('describes studioMedia test behavior 1', () => {
  it('verifies studioMedia test behavior 2', async () => {
    const blob = new Blob(['image'], { type: 'image/png' });
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, blob: async () => blob });
    vi.stubGlobal('fetch', fetchMock);
    const view = render(<MediaPartView part={{ type: 'image', media }} />);
    expect(screen.getByLabelText('Loading media')).toBeInTheDocument();
    const image = await screen.findByAltText('Attached image');
    const [url, options] = fetchMock.mock.calls[0];
    expect(url).toBe('https://studio.example/api/v1/media/protected-media');
    expect(options.headers.get('Authorization')).toBe('Bearer private-token');
    expect(options.signal.aborted).toBe(false);
    expect(createObjectURL).toHaveBeenCalledWith(blob);
    expect(image).toHaveAttribute('src', 'blob:protected');
    view.unmount();
    expect(options.signal.aborted).toBe(true);
    expect(revokeObjectURL).toHaveBeenCalledWith('blob:protected');
  });

  it('verifies studioMedia test behavior 3', async () => {
    let resolveBlob!: (blob: Blob) => void;
    const fetchMedia = vi.spyOn(mediaApi, 'fetchMedia').mockReturnValue(new Promise((resolve) => { resolveBlob = resolve; }));
    const view = render(<MediaPartView part={{ type: 'image', media }} />);
    view.unmount();
    await act(async () => resolveBlob(new Blob(['late'])));
    expect(fetchMedia.mock.calls[0][1]?.aborted).toBe(true);
    expect(createObjectURL).not.toHaveBeenCalled();
  });
});

describe('describes studioMedia test behavior 4', () => {
  it('verifies studioMedia test behavior 5', async () => {
    const view = render(<VideoWithFirstFrame src="blob:video" controls />);
    const visibleVideo = view.container.querySelector('video')!;
    const load = vi.mocked(HTMLMediaElement.prototype.load);
    const decoder = load.mock.contexts[0] as HTMLVideoElement;
    setVideoMetadata(decoder);
    const drawImage = vi.fn();
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({ drawImage } as unknown as CanvasRenderingContext2D);
    const posterBlob = new Blob(['poster'], { type: 'image/jpeg' });
    const toBlob = vi.spyOn(HTMLCanvasElement.prototype, 'toBlob').mockImplementation((callback) => callback(posterBlob));
    await act(async () => { decoder.dispatchEvent(new Event('loadeddata')); });
    expect(drawImage).toHaveBeenCalledWith(decoder, 0, 0, 1280, 720);
    expect(toBlob).toHaveBeenCalledWith(expect.any(Function), 'image/jpeg', 0.85);
    expect(createObjectURL).toHaveBeenCalledWith(posterBlob);
    expect(visibleVideo).toHaveAttribute('poster', 'blob:protected');
    expect(visibleVideo).toHaveAttribute('controls');
    view.unmount();
    expect(decoder.onloadeddata).toBeNull();
    expect(decoder.hasAttribute('src')).toBe(false);
    expect(HTMLMediaElement.prototype.pause).toHaveBeenCalled();
    expect(revokeObjectURL).toHaveBeenCalledWith('blob:protected');
  });

  it('verifies studioMedia test behavior 6', async () => {
    vi.spyOn(mediaApi, 'fetchMedia').mockResolvedValue(new Blob(['video']));
    const view = render(<MediaPartView part={{ type: 'video', media }} />);
    await waitFor(() => expect(view.container.querySelector('video')).toHaveAttribute('src', 'blob:protected'));
    expect(view.container.querySelector('video')).toHaveAttribute('controls');
    expect(view.container.querySelector('video')).toHaveClass('message-media', 'media-video');
  });

  it('verifies studioMedia test behavior 7', () => {
    const composer = readFileSync(resolve('src/features/chat/components/ChatComposer.tsx'), 'utf8');
    expect(composer).toContain('<VideoWithFirstFrame muted src={attachment.previewUrl} />');
  });
});

describe('describes studioMedia test behavior 8', () => {
  it('verifies studioMedia test behavior 9', async () => {
    const { result } = renderHook(() => useChatAttachments('session', 0, vi.fn()));
    const file = new File(['video'], 'clip.mp4', { type: 'video/mp4' });
    act(() => result.current.selectAttachments([file] as unknown as FileList));
    const decoder = captureDecoder();
    const upload = vi.spyOn(mediaApi, 'uploadMedia').mockResolvedValue({ media, created_at: '' });
    const pending = result.current.uploadAttachments();
    expect(upload).not.toHaveBeenCalled();
    expect(HTMLMediaElement.prototype.load).toHaveBeenCalled();
    setVideoMetadata(decoder);
    decoder.dispatchEvent(new Event('loadedmetadata'));
    await expect(pending).resolves.toEqual([{ type: 'video', media, width: 1920, height: 1080, duration_ms: 1250 }]);
    expect(upload).toHaveBeenCalledExactlyOnceWith(file);
    expect(decoder.onloadedmetadata).toBeNull();
    expect(decoder.hasAttribute('src')).toBe(false);
    expect(revokeObjectURL).toHaveBeenCalledWith('blob:protected');
  });

  it.each(['invalid', 'error', 'timeout'] as const)('verifies studioMedia test behavior 10', async (failure) => {
    vi.useFakeTimers();
    const { result } = renderHook(() => useChatAttachments('session', 0, vi.fn()));
    const file = new File(['video'], 'bad.mp4', { type: 'video/mp4' });
    act(() => result.current.selectAttachments([file] as unknown as FileList));
    const decoder = captureDecoder();
    const upload = vi.spyOn(mediaApi, 'uploadMedia');
    const rejected = expect(result.current.uploadAttachments()).rejects.toThrow('Unable to read video metadata');
    if (failure === 'timeout') {
      await vi.advanceTimersByTimeAsync(15_000);
    } else if (failure === 'error') {
      decoder.dispatchEvent(new Event('error'));
    } else {
      setVideoMetadata(decoder, 0, 1080, Number.NaN);
      decoder.dispatchEvent(new Event('loadedmetadata'));
    }
    await rejected;
    expect(upload).not.toHaveBeenCalled();
    expect(decoder.onerror).toBeNull();
    expect(decoder.onloadedmetadata).toBeNull();
    expect(revokeObjectURL).toHaveBeenCalledWith('blob:protected');
    expect(vi.getTimerCount()).toBe(0);
  });

  it('verifies studioMedia test behavior 11', async () => {
    const decoder = captureDecoder();
    const pending = mediaMetadata(new File(['video'], 'clip.mp4'), 'video');
    setVideoMetadata(decoder, 640, 480, 2);
    decoder.dispatchEvent(new Event('loadedmetadata'));
    await expect(pending).resolves.toEqual({ width: 640, height: 480, duration_ms: 2000 });
    expect(revokeObjectURL).toHaveBeenCalledExactlyOnceWith('blob:protected');
  });
});

describe('describes studioMedia test behavior 12', () => {
  it('verifies studioMedia test behavior 13', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => {});
    vi.spyOn(mediaApi, 'fetchMedia').mockRejectedValue(new Error('forbidden'));
    render(<MediaPartView part={{ type: 'image', media }} />);
    expect(await screen.findByRole('alert')).toHaveTextContent('Unable to load attachment');
    expect(screen.queryByAltText('Attached image')).not.toBeInTheDocument();
  });

  it('verifies studioMedia test behavior 14', async () => {
    vi.useFakeTimers();
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, blob: async () => new Blob(['document']) });
    vi.stubGlobal('fetch', fetchMock);
    let anchor: HTMLAnchorElement | undefined;
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) {
      anchor = this;
      expect(document.body.contains(this)).toBe(true);
      expect(this.download).toBe('notes.md');
      expect(this.href).toBe('blob:protected');
    });
    render(<DocumentPartView part={{ type: 'document', media, name: 'notes.md' }} />);
    const button = screen.getByRole('button');
    fireEvent.click(button);
    expect(button).toBeDisabled();
    await act(async () => { await Promise.resolve(); });
    expect(fetchMock.mock.calls[0][0]).toBe('https://studio.example/api/v1/media/protected-media');
    expect(fetchMock.mock.calls[0][1].headers.get('Authorization')).toBe('Bearer private-token');
    expect(anchor).toBeDefined();
    expect(document.body.contains(anchor!)).toBe(false);
    expect(button).toBeEnabled();
    expect(revokeObjectURL).not.toHaveBeenCalled();
    await act(() => vi.advanceTimersByTimeAsync(1000));
    expect(revokeObjectURL).toHaveBeenCalledExactlyOnceWith('blob:protected');
  });

  it('verifies studioMedia test behavior 15', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => {});
    const fetchMedia = vi.spyOn(mediaApi, 'fetchMedia').mockRejectedValue(new Error('offline'));
    render(<DocumentPartView part={{ type: 'document', media, name: 'notes.md' }} />);
    fireEvent.click(screen.getByRole('button'));
    expect(await screen.findByText('Download failed — click to retry')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button'));
    await waitFor(() => expect(fetchMedia).toHaveBeenCalledTimes(2));
  });
});
