/** 跟随聊天内容高度变化，用户向上阅读时暂停跟随，避免抢夺滚动位置。 */
import { useCallback, useEffect, useRef, useState } from 'react';

/** 管理聊天滚动容器；ResizeObserver 同时覆盖流式渲染、公式和媒体加载。 */
export function useChatAutoScroll(sessionId: string | null, enabled: boolean) {
  const scrollerRef = useRef<HTMLDivElement | null>(null);
  const followingRef = useRef(true);
  const [following, setFollowing] = useState(true);

  /** 用户滚动后根据与底部的距离切换跟随状态，保留历史阅读位置。 */
  const handleScroll = useCallback(() => {
    const scroller = scrollerRef.current;
    if (!scroller) return;
    const next = scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight <= 80;
    followingRef.current = next;
    setFollowing(next);
  }, []);

  /** 响应回到底部操作并重新开启后续输出跟随。 */
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
