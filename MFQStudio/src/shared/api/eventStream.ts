/** 解析分块 SSE 字节流，统一处理换行、取消和读取器资源释放。 */

/** 消费 JSON SSE 事件；缺失结束符、非法 JSON 或读取取消时向调用方抛错。 */
export async function readEventStream<T>(
  response: Response,
  onEvent: (event: T) => void,
  signal?: AbortSignal,
): Promise<void> {
  if (!response.body || !response.headers.get('content-type')?.includes('text/event-stream')) {
    throw new Error('MFQ Server returned an invalid streaming response');
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let line = '';
  let data: string[] = [];
  let afterCarriageReturn = false;
  let ended = false;
  const finishLine = () => {
    if (!line) {
      if (data.length) {
        const payload = data.join('\n');
        data = [];
        if (payload) onEvent(JSON.parse(payload) as T);
      }
    } else if (line === 'data' || line.startsWith('data:')) {
      const value = line === 'data' ? '' : line.slice(5);
      data.push(value.startsWith(' ') ? value.slice(1) : value);
    }
    line = '';
  };
  const consume = (text: string) => {
    for (const character of text) {
      if (afterCarriageReturn && character === '\n') {
        afterCarriageReturn = false;
        continue;
      }
      afterCarriageReturn = character === '\r';
      if (character === '\r' || character === '\n') finishLine();
      else line += character;
    }
  };
  const cancel = () => {
    void reader.cancel(signal?.reason).catch(() => undefined);
  };
  signal?.addEventListener('abort', cancel, { once: true });
  try {
    signal?.throwIfAborted();
    for (;;) {
      const { value, done } = await reader.read();
      signal?.throwIfAborted();
      consume(decoder.decode(value, { stream: !done }));
      if (done) {
        ended = true;
        break;
      }
    }
    if (data.length || line.startsWith('data:') || line === 'data') {
      throw new Error('MFQ Server stream ended with an incomplete event');
    }
  } finally {
    signal?.removeEventListener('abort', cancel);
    if (!ended) await reader.cancel().catch(() => undefined);
    reader.releaseLock();
  }
}
