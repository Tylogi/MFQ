import { useCallback, useEffect, useRef, useState } from 'react';
import { runtimeApi, type PrefixCacheInventory, type PrefixCacheText } from '../../shared/api/resources/runtime';
import { useRuntime } from '../../app/RuntimeProvider';
import { useConnectionScope } from '../../app/useConnectionScope';
import { useSettings } from '../settings/SettingsProvider';
import { SectionLabel, TMPanel } from '../../app/display';
import { errorMessage, formatBytes, formatNumber } from '../../app/formatters';
import { studioConfirm } from '../../studio';
import { toast } from '../../stores/toastStore';

export function PrefixCacheInventoryPanel() {
  const { connectionRevision } = useRuntime();
  return <CacheInventory key={connectionRevision} />;
}

function CacheInventory() {
  const { tr } = useSettings();
  const { refreshRuntime } = useRuntime();
  const connectionScope = useConnectionScope();
  const [inventory, setInventory] = useState<PrefixCacheInventory | null>(null);
  const [detail, setDetail] = useState<PrefixCacheInventory | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [text, setText] = useState<PrefixCacheText | null>(null);
  const [textBlock, setTextBlock] = useState<string | null>(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [reading, setReading] = useState(false);
  const clearing = useRef(false);
  const sequence = useRef(0);
  const inventorySequence = useRef(0);
  const refresh = useCallback(async (signal?: AbortSignal) => {
    const current = connectionScope();
    const version = ++inventorySequence.current;
    try {
      const value = await runtimeApi.prefixCacheEntries(undefined, 0, signal);
      if (signal?.aborted || !current() || version !== inventorySequence.current) return;
      setInventory(value);
      setError('');
    } catch (cause) {
      if (!signal?.aborted && current() && version === inventorySequence.current) setError(errorMessage(cause));
    }
  }, [connectionScope]);
  useEffect(() => {
    const controller = new AbortController();
    let pending = false;
    async function update() {
      if (pending || document.hidden) return;
      pending = true;
      try { await refresh(controller.signal); } finally { pending = false; }
    }
    void update();
    const timer = window.setInterval(() => void update(), 5000);
    document.addEventListener('visibilitychange', update);
    return () => { controller.abort(); window.clearInterval(timer); document.removeEventListener('visibilitychange', update); sequence.current += 1; };
  }, [refresh]);
  async function inspect(namespace: string, offset = 0) {
    const current = connectionScope();
    const version = ++sequence.current;
    setSelected(namespace);
    setDetail(null);
    setText(null);
    setTextBlock(null);
    setReading(true);
    try {
      const value = await runtimeApi.prefixCacheEntries(namespace, offset);
      if (version === sequence.current && current()) setDetail(value);
    } catch (cause) { if (version === sequence.current && current()) toast.error(errorMessage(cause)); }
    finally { if (version === sequence.current && current()) setReading(false); }
  }
  async function read(block: string, offset = 0) {
    if (!selected) return;
    const current = connectionScope();
    const version = ++sequence.current;
    setTextBlock(block);
    setText(null);
    setReading(true);
    try {
      const value = await runtimeApi.prefixCacheText(selected, block, offset);
      if (version === sequence.current && current()) setText(value);
    } catch (cause) { if (version === sequence.current && current()) toast.error(errorMessage(cause)); }
    finally { if (version === sequence.current && current()) setReading(false); }
  }
  async function clear(namespace?: string) {
    const current = connectionScope();
    if (!inventory?.can_clear || clearing.current) return;
    const group = inventory.data.find((item) => item.id === namespace);
    const message = namespace
      ? tr(`清理此缓存组（${formatBytes(group?.bytes ?? 0)}）？模型文件和聊天记录不会删除。`,
          `Clear this cache group (${formatBytes(group?.bytes ?? 0)})? Model files and chat history will be kept.`)
      : tr(`清理全部前缀缓存（${formatBytes(inventory.total_bytes)}），包括未载入模型的 SSD 缓存与 RAM 热缓存？模型文件和聊天记录不会删除。`,
          `Clear all prefix caches (${formatBytes(inventory.total_bytes)}), including unloaded models’ SSD caches and RAM hot caches? Model files and chat history will be kept.`);
    clearing.current = true;
    setBusy(true);
    try {
      if (!await studioConfirm(message) || !current()) return;
      const result = await runtimeApi.purgePrefixCache(namespace);
      if (!current()) return;
      sequence.current += 1;
      setSelected(null); setDetail(null); setText(null); setTextBlock(null); setReading(false);
      await Promise.all([refresh(), refreshRuntime(false)]);
      if (!current()) return;
      const summary = tr(`已清理 ${formatNumber(result.removed_blocks)} 个块，释放 ${formatBytes(result.released_bytes)}`,
        `Cleared ${formatNumber(result.removed_blocks)} blocks; released ${formatBytes(result.released_bytes)}`);
      if (result.failed_blocks) toast.error(`${summary} · ${tr('部分块未能删除', 'Some blocks could not be removed')}`);
      else toast.success(summary);
    } catch (cause) { if (current()) toast.error(errorMessage(cause)); }
    finally { clearing.current = false; if (current()) setBusy(false); }
  }
  const reasons: Record<string, string> = {
    tokenizer_unavailable: tr('对应模型或 tokenizer 不可用', 'The model or tokenizer is unavailable'),
    text_invalid: tr('文本记录校验失败', 'Text record integrity check failed'),
    incomplete_chain: tr('前序缓存块已回收', 'Earlier prefix blocks have been reclaimed'),
    cache_not_found: tr('缓存块已被回收或清理', 'This cache block has been reclaimed or cleared'),
    invalid_cache_id: tr('缓存标识无效', 'Invalid cache identifier'),
    prefix_too_large: tr('此前缀过长，无法显示文本', 'This prefix is too large to display'),
    text_not_saved: tr('旧缓存未保存文本记录', 'This older cache has no saved text record'),
  };
  const reason = reasons[text?.reason || ''] ?? tr('缓存文本暂不可用', 'Cache text is unavailable');
  return <>
    <SectionLabel title={tr('缓存管理', 'Cache management')} />
    <TMPanel className="prefix-inventory-panel">
      <div className="panel-heading">
        <div><h2>{tr('SSD 缓存内容', 'SSD cache contents')}</h2><p className="prefix-cache-directory">{inventory?.directory || '--'}</p></div>
        <div className="prefix-cache-actions">
          <button type="button" disabled={busy} onClick={() => void refresh()}>{tr('刷新', 'Refresh')}</button>
          <button type="button" className="danger" disabled={busy || !inventory?.can_clear}
            onClick={() => void clear()}>{busy ? tr('清理中…', 'Clearing…') : tr('一键清理', 'Clear all')}</button>
        </div>
      </div>
      {error && <p className="resource-monitor-error">{error}</p>}
      {inventory && !inventory.can_clear && <p className="prefix-cache-note">{tr('推理或缓存写入结束后可清理', 'Clearing is available after inference and cache writes finish')}</p>}
      {inventory?.data.length === 0 && <p className="prefix-cache-note">{tr('暂无 SSD 缓存', 'No SSD caches')}</p>}
      {inventory?.data.map((group) => <div className="prefix-cache-group" key={group.id}>
        <div><strong>{group.model_name || tr(`旧缓存 · ${group.id.slice(0, 10)}`, `Older cache · ${group.id.slice(0, 10)}`)}</strong>
          <small>{group.context_size ? `${formatNumber(group.context_size)} ctx · ` : ''}{formatNumber(group.blocks)} {tr('块', 'blocks')} · {formatBytes(group.bytes)}</small>
          <small>{tr('最近使用', 'Last used')} {new Date(group.last_used_at).toLocaleString()}{group.text_blocks === 0 ? ` · ${tr('未保存文本', 'No saved text')}` : ''}</small>
        </div>
        <div className="prefix-cache-actions"><button type="button" disabled={busy || reading} onClick={() => void inspect(group.id)}>{tr('查看', 'Inspect')}</button>
          <button type="button" disabled={busy || !inventory.can_clear} onClick={() => void clear(group.id)}>{tr('清理', 'Clear')}</button></div>
      </div>)}
      {selected && detail && <div className="prefix-cache-detail">
        <div className="prefix-cache-detail-heading"><strong>{tr('缓存块', 'Cache blocks')}</strong>
          <button type="button" onClick={() => { sequence.current += 1; setSelected(null); setDetail(null); setText(null); setReading(false); }}>{tr('关闭', 'Close')}</button></div>
        <p className="prefix-cache-note">{detail.data[0]?.codec || '--'} · {detail.data[0]?.architecture || '--'} · {tr('最长完整前缀', 'Longest complete prefix')} {formatNumber(detail.data[0]?.max_prefix_tokens)} tokens</p>
        {detail.blocks.map((block) => <div className="prefix-cache-block" key={block.id}>
          <span title={block.id}>{block.id.slice(0, 10)}</span><span>{formatNumber(block.prefix_tokens)} tokens{!block.complete_chain ? ` · ${tr('前序已回收', 'Earlier blocks reclaimed')}` : ''}</span>
          <span>{formatBytes(block.bytes)}</span><button type="button" disabled={reading || !block.text_available}
            onClick={() => void read(block.id)}>{tr('查看文本', 'View text')}</button>
        </div>)}
        <div className="prefix-cache-actions"><button type="button" disabled={reading || detail.offset === 0} onClick={() => void inspect(selected, Math.max(0, detail.offset - detail.limit))}>{tr('上一页', 'Previous')}</button>
          <button type="button" disabled={reading || detail.offset + detail.blocks.length >= (detail.data[0]?.blocks ?? 0)} onClick={() => void inspect(selected, detail.offset + detail.limit)}>{tr('下一页', 'Next')}</button></div>
        {text && <div className="prefix-cache-text"><strong>{tr('缓存文本', 'Cached text')}</strong>
          {text.available ? <><pre>{text.text}</pre><div className="prefix-cache-actions">
            <button type="button" disabled={reading || !text.offset} onClick={() => textBlock && void read(textBlock, Math.max(0, (text.offset ?? 0) - 8192))}>{tr('上一段', 'Previous section')}</button>
            <small>{formatNumber(text.total_tokens)} tokens</small>
            <button type="button" disabled={reading || text.next_offset == null} onClick={() => textBlock && void read(textBlock, text.next_offset ?? 0)}>{tr('下一段', 'Next section')}</button></div></>
            : <p className="prefix-cache-note">{reason}</p>}
        </div>}
      </div>}
    </TMPanel>
  </>;
}
