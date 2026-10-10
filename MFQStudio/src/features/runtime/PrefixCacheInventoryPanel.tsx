import { useCallback, useEffect, useRef, useState } from 'react';
import { runtimeApi, type PrefixCacheInventory, type PrefixCacheText, type PrefixCacheFilters } from '../../shared/api/resources/runtime';
import { ListToolbar } from '../../shared/ui/ListToolbar';
import { useRuntime } from '../../app/RuntimeProvider';
import { hasSeparateVram } from './memoryArchitecture';
import { useConnectionScope } from '../../app/useConnectionScope';
import { useSettings } from '../settings/SettingsProvider';
import { Icon, SectionLabel, TMPanel } from '../../app/display';
import { errorMessage, formatBytes, formatNumber } from '../../app/formatters';
import { studioConfirm } from '../../studio';
import { toast } from '../../stores/toastStore';

const defaultFilters: PrefixCacheFilters = { query: '', search_in: 'all', text_filter: 'all', chain_filter: 'all', sort: 'recent' };

export function PrefixCacheInventoryPanel() {
  const { connectionRevision } = useRuntime();
  return <CacheInventory key={connectionRevision} />;
}

function CacheInventory() {
  const { tr } = useSettings();
  const { refreshRuntime, runtime } = useRuntime();
  const connectionScope = useConnectionScope();
  const [inventory, setInventory] = useState<PrefixCacheInventory | null>(null);
  const [detail, setDetail] = useState<PrefixCacheInventory | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [text, setText] = useState<PrefixCacheText | null>(null);
  const [textBlock, setTextBlock] = useState<string | null>(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [reading, setReading] = useState(false);
  const [filters, setFilters] = useState<PrefixCacheFilters>(defaultFilters);
  const searchTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const detailRequest = useRef<AbortController | null>(null);
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
    return () => { controller.abort(); detailRequest.current?.abort(); clearTimeout(searchTimer.current); window.clearInterval(timer); document.removeEventListener('visibilitychange', update); sequence.current += 1; };
  }, [refresh]);
  async function inspect(namespace: string, offset = 0, nextFilters = namespace === selected ? filters : defaultFilters) {
    clearTimeout(searchTimer.current);
    detailRequest.current?.abort();
    const controller = new AbortController();
    detailRequest.current = controller;
    const current = connectionScope();
    const version = ++sequence.current;
    setSelected(namespace);
    setFilters(nextFilters);
    setDetail(null);
    setText(null);
    setTextBlock(null);
    setReading(true);
    try {
      const value = await runtimeApi.prefixCacheEntries(namespace, offset, controller.signal, nextFilters);
      if (version === sequence.current && current()) setDetail(value);
    } catch (cause) { if (version === sequence.current && current()) toast.error(errorMessage(cause)); }
    finally { if (version === sequence.current && current()) setReading(false); }
  }
  function updateFilters(next: PrefixCacheFilters) {
    setFilters(next);
    clearTimeout(searchTimer.current);
    detailRequest.current?.abort();
    sequence.current += 1;
    setDetail(null); setText(null); setTextBlock(null);
    if (selected) {
      setReading(true);
      searchTimer.current = setTimeout(() => void inspect(selected, 0, next), 250);
    }
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
      : hasSeparateVram(runtime) ? tr(`清理全部前缀缓存（${formatBytes(inventory.total_bytes)}），包括未载入模型的 SSD 缓存与显存热缓存？模型文件和聊天记录不会删除。`,
          `Clear all prefix caches (${formatBytes(inventory.total_bytes)}), including unloaded models’ SSD caches and VRAM hot caches? Model files and chat history will be kept.`) : tr(`清理全部前缀缓存（${formatBytes(inventory.total_bytes)}），包括未载入模型的 SSD 缓存与 RAM 热缓存？模型文件和聊天记录不会删除。`,
          `Clear all prefix caches (${formatBytes(inventory.total_bytes)}), including unloaded models’ SSD caches and RAM hot caches? Model files and chat history will be kept.`);
    clearing.current = true;
    setBusy(true);
    try {
      if (!await studioConfirm(message) || !current()) return;
      const result = await runtimeApi.purgePrefixCache(namespace);
      if (!current()) return;
      sequence.current += 1;
      clearTimeout(searchTimer.current);
      detailRequest.current?.abort();
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
      <div className="panel-heading prefix-inventory-heading">
        <h2>{tr('SSD 缓存内容', 'SSD cache contents')}</h2>
        <div className="prefix-cache-actions">
          <button type="button" disabled={busy} onClick={() => void refresh()}><Icon name="refresh" size={14} />{tr('刷新', 'Refresh')}</button>
          <button type="button" className="danger" disabled={busy || !inventory?.can_clear}
            onClick={() => void clear()}><Icon name="trash" size={14} />{busy ? tr('清理中…', 'Clearing…') : tr('一键清理', 'Clear all')}</button>
        </div>
      </div>
      <div className="prefix-cache-directory"><Icon name="folder" size={15} />
        <div><span>{tr('缓存目录', 'Cache directory')}</span><p>{inventory?.directory || '--'}</p></div>
      </div>
      <div className="cache-stats prefix-inventory-summary">
        <div><span>{tr('SSD 占用', 'SSD usage')}</span><strong>{inventory ? inventory.total_bytes === 0 ? '0 B' : formatBytes(inventory.total_bytes) : '--'}</strong></div>
        <div><span>{tr('缓存块', 'Cache blocks')}</span><strong>{inventory ? formatNumber(inventory.total_blocks) : '--'}</strong></div>
        <div><span>{tr('缓存组', 'Cache groups')}</span><strong>{inventory ? formatNumber(inventory.data.length) : '--'}</strong></div>
      </div>
      {error && <p className="resource-monitor-error">{error}</p>}
      {inventory && !inventory.can_clear && <p className="prefix-cache-note">{tr('推理或缓存写入结束后可清理', 'Clearing is available after inference and cache writes finish')}</p>}
      <div className="prefix-cache-list">
        {!inventory && !error && <p className="prefix-cache-empty">{tr('正在读取缓存…', 'Reading caches…')}</p>}
        {inventory?.data.length === 0 && <p className="prefix-cache-empty">{tr('暂无 SSD 缓存', 'No SSD caches')}</p>}
        {inventory?.data.map((group) => <div className={`prefix-cache-group${selected === group.id ? ' selected' : ''}`} key={group.id}>
          <div className="prefix-cache-group-info">
            <strong title={group.model_name || group.id}>{group.model_name || tr(`旧缓存 · ${group.id.slice(0, 10)}`, `Older cache · ${group.id.slice(0, 10)}`)}</strong>
            <div className="prefix-cache-group-metrics">
              <span>{formatBytes(group.bytes)}</span><span>{formatNumber(group.blocks)} {tr('块', 'blocks')}</span>
              {group.context_size ? <span>{formatNumber(group.context_size)} ctx</span> : null}
            </div>
            <small>{tr('最近使用', 'Last used')} {new Date(group.last_used_at).toLocaleString()}{group.text_blocks === 0 ? ` · ${tr('未保存文本', 'No saved text')}` : ''}</small>
          </div>
          <div className="prefix-cache-actions"><button type="button" disabled={busy || reading} onClick={() => void inspect(group.id)}><Icon name="search" size={14} />{tr('查看', 'Inspect')}</button>
            <button type="button" className="danger" disabled={busy || !inventory.can_clear} onClick={() => void clear(group.id)}><Icon name="trash" size={14} />{tr('清理', 'Clear')}</button></div>
        </div>)}
      </div>
      {selected && <div className="prefix-cache-detail">
        <div className="prefix-cache-detail-heading"><strong>{tr('缓存块详情', 'Cache block details')}</strong>
          <button type="button" onClick={() => { sequence.current += 1; clearTimeout(searchTimer.current); detailRequest.current?.abort(); setSelected(null); setDetail(null); setText(null); setReading(false); }}>{tr('关闭', 'Close')}</button></div>
        <ListToolbar label={tr('搜索缓存块', 'Search cache blocks')} placeholder={tr('搜索标识或块内文本', 'Search ID or block text')}
          query={filters.query} onQueryChange={query => updateFilters({ ...filters, query })}
          count={detail?.matched_blocks ?? detail?.data[0]?.blocks ?? 0}
          total={inventory?.data.find(group => group.id === selected)?.blocks ?? 0}
          activeFilters={[filters.search_in, filters.text_filter, filters.chain_filter].filter(value => value !== 'all').length}
          onReset={() => updateFilters(defaultFilters)}>
          <label>{tr('搜索范围', 'Search in')}<select aria-label={tr('缓存搜索范围', 'Cache search scope')} value={filters.search_in} onChange={event => updateFilters({ ...filters, search_in: event.target.value as PrefixCacheFilters['search_in'] })}>
            <option value="all">{tr('标识与文本', 'ID and text')}</option><option value="text">{tr('仅文本', 'Text only')}</option><option value="id">{tr('仅标识', 'ID only')}</option>
          </select></label>
          <label>{tr('文本记录', 'Text records')}<select aria-label={tr('缓存文本记录', 'Cache text records')} value={filters.text_filter} onChange={event => updateFilters({ ...filters, text_filter: event.target.value as PrefixCacheFilters['text_filter'] })}>
            <option value="all">{tr('全部', 'All')}</option><option value="saved">{tr('已保存文本', 'Text saved')}</option><option value="missing">{tr('未保存文本', 'No saved text')}</option>
          </select></label>
          <label>{tr('前缀链', 'Prefix chain')}<select aria-label={tr('缓存前缀链', 'Cache prefix chain')} value={filters.chain_filter} onChange={event => updateFilters({ ...filters, chain_filter: event.target.value as PrefixCacheFilters['chain_filter'] })}>
            <option value="all">{tr('全部', 'All')}</option><option value="complete">{tr('完整', 'Complete')}</option><option value="incomplete">{tr('前序已回收', 'Earlier blocks reclaimed')}</option>
          </select></label>
          <label>{tr('排序', 'Sort')}<select aria-label={tr('缓存块排序', 'Cache block sort')} value={filters.sort} onChange={event => updateFilters({ ...filters, sort: event.target.value as PrefixCacheFilters['sort'] })}>
            <option value="recent">{tr('最近使用', 'Recently used')}</option><option value="oldest">{tr('最早使用', 'Oldest first')}</option><option value="length">{tr('前缀长度', 'Prefix length')}</option><option value="size">{tr('占用大小', 'Size')}</option>
          </select></label>
        </ListToolbar>
        {!detail && reading && <p className="prefix-cache-note" role="status">{tr('正在搜索缓存块…', 'Searching cache blocks…')}</p>}
        {detail?.search_text_unavailable && <p className="prefix-cache-note" role="status">{tr('对应 tokenizer 不可用，部分文本无法搜索；仍可按标识筛选。', 'The tokenizer is unavailable; some text cannot be searched. ID search is still available.')}</p>}
        {detail && <>
        <p className="prefix-cache-note">{detail.data[0]?.codec || '--'} · {detail.data[0]?.architecture || '--'} · {tr('最长完整前缀', 'Longest complete prefix')} {formatNumber(detail.data[0]?.max_prefix_tokens)} tokens</p>
        <div className="prefix-cache-blocks">
          <div className="prefix-cache-block prefix-cache-block-header"><span>{tr('标识', 'ID')}</span><span>{tr('前缀长度', 'Prefix length')}</span><span>{tr('占用', 'Size')}</span><span>{tr('文本', 'Text')}</span></div>
          {detail.blocks.length === 0 && <p className="prefix-cache-empty">{tr('没有匹配的缓存块', 'No matching cache blocks')}</p>}
          {detail.blocks.map((block) => <div className="prefix-cache-block" key={block.id}>
            <span className="prefix-cache-block-id" title={block.id}>{block.id.slice(0, 10)}</span><span>{formatNumber(block.prefix_tokens)} tokens{!block.complete_chain ? ` · ${tr('前序已回收', 'Earlier blocks reclaimed')}` : ''}</span>
            <span>{formatBytes(block.bytes)}</span><button type="button" disabled={reading || !block.text_available}
              onClick={() => void read(block.id, filters.query.trim() && filters.search_in !== 'id' ? Math.max(0, block.prefix_tokens - block.tokens) : 0)}>{tr('查看文本', 'View text')}</button>
          </div>)}
        </div>
        <div className="prefix-cache-actions prefix-cache-pagination"><small>{formatNumber(detail.offset + (detail.blocks.length ? 1 : 0))}–{formatNumber(detail.offset + detail.blocks.length)} / {formatNumber(detail.matched_blocks ?? detail.data[0]?.blocks)}</small>
          <button type="button" disabled={reading || detail.offset === 0} onClick={() => void inspect(selected, Math.max(0, detail.offset - detail.limit))}>{tr('上一页', 'Previous')}</button>
          <button type="button" disabled={reading || detail.offset + detail.blocks.length >= (detail.matched_blocks ?? detail.data[0]?.blocks ?? 0)} onClick={() => void inspect(selected, detail.offset + detail.limit)}>{tr('下一页', 'Next')}</button></div>
        {reading && textBlock && <p className="prefix-cache-note" role="status">{tr('正在读取文本…', 'Reading text…')}</p>}
        {text && <div className="prefix-cache-text"><strong>{tr('缓存文本', 'Cached text')}</strong>
          {text.available ? <><pre>{text.text}</pre><div className="prefix-cache-actions prefix-cache-pagination">
            <small>{formatNumber(text.total_tokens)} tokens</small>
            <button type="button" disabled={reading || !text.offset} onClick={() => textBlock && void read(textBlock, Math.max(0, (text.offset ?? 0) - 8192))}>{tr('上一段', 'Previous section')}</button>
            <button type="button" disabled={reading || text.next_offset == null} onClick={() => textBlock && void read(textBlock, text.next_offset ?? 0)}>{tr('下一段', 'Next section')}</button></div></>
            : <p className="prefix-cache-note">{reason}</p>}
        </div>}
        </>}
      </div>}
    </TMPanel>
  </>;
}
