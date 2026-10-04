/** Follow chat content height changes, pausing when the user scrolls up to read so their position is not overridden. */
import { useCallback, useEffect, useRef, useState } from 'react';
/** Manage the chat scroll container; ResizeObserver covers streaming renders, formulas, and media loading. */
export function useChatAutoScroll(sessionId: string | null, enabled: boolean) {
  const scrollerRef = useRef<HTMLDivElement | null>(null);
  const followingRef = useRef(true);
  const [following, setFollowing] = useState(true);
/** Update follow state based on distance from the bottom after user scrolling, preserving history reading position. */
  const handleScroll = useCallback(() => {
    const scroller = scrollerRef.current;
    if (!scroller) return;
    const next = scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight <= 80;
    followingRef.current = next;
    setFollowing(next);
  }, []);
/** Handle the return-to-bottom action and re-enable following for subsequent output. */
  const scrollToBottom = useCallback(() => {
    followingRef.current = true;
    setFollowing(true);
    const scroller = scrollerRef.current;
    if (scroller) scroller.scrollTop = scroller.scrollHeight;
  }, []);

  useEffect(() => {
    if (!enabled) return;
    const scroller = scrollerRef.current;
    if (!scroller) return;
    followingRef.current = true;
    setFollowing(true);
    let frame: number | null = null;
    const follow = () => {
      if (!followingRef.current || frame !== null) return;
      frame = requestAnimationFrame(() => {
        frame = null;
        if (followingRef.current) scroller.scrollTop = scroller.scrollHeight;
      });
    };
    const observer = new ResizeObserver(follow);
    observer.observe(scroller);
    if (scroller.firstElementChild) observer.observe(scroller.firstElementChild);
    follow();
    return () => {
      observer.disconnect();
      if (frame !== null) cancelAnimationFrame(frame);
    };
  }, [sessionId, enabled]);

  return { scrollerRef, following, handleScroll, scrollToBottom };
}
