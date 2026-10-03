import { FormEvent, type ReactNode, useEffect, useMemo, useRef, useState } from "react";

import type {
  HubModelInfo,
  HubModelSummary,
  HubModelVariant,
  HubSystemProfile,
  JobKindResource,
  JobResource,
  ModelConfigurationStatus,
  OfficialModelInfo,
  OfficialModelList,
  OfficialModelSource,
} from '../../shared/api/types';
import { jobsApi } from '../../shared/api/resources/jobs';
import { modelsApi } from '../../shared/api/resources/models';
import { openStudioExternal } from '../../shared/platform/studio';
import { BackendBadge } from './BackendBadge';

type Translate = (chinese: string, english: string) => string;
export type ModelBrowserTab = 'official' | 'community' | 'downloads';
export type DownloadOrigin = { x: number; y: number };

interface ModelBrowserProps {
  jobKinds: JobKindResource[];
  onError(message: string): void;
  onJobCreated(job: JobResource, origin: DownloadOrigin): void;
  tab: ModelBrowserTab;
  onTabChange(tab: ModelBrowserTab): void;
  downloadQueue?: ReactNode;
  tr: Translate;
}

const SUPPORT_FILES = [
  "*.json",
  "*.txt",
  "*.model",
  "*.tiktoken",
  "*.jinja",
  "tokenizer*",
  "processor*",
  "preprocessor*",
];

function formatBytes(value?: number | null): string {
  if (!value) return "—";
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let amount = value;
  let unit = 0;
  while (amount >= 1024 && unit < units.length - 1) {
    amount /= 1024;
    unit += 1;
  }
  return `${amount.toFixed(unit < 2 ? 0 : 1)} ${units[unit]}`;
}

function formatCount(value: number): string {
  return new Intl.NumberFormat(undefined, { notation: "compact" }).format(value);
}

function hardwareSummary(system: HubSystemProfile, tr: Translate): string {
  const ram = system.physical_memory_bytes ? `${formatBytes(system.physical_memory_bytes).replace(/\.0 /, ' ')} RAM` : null;
  const gpu = system.gpu_names?.join(' + ');
  if (system.backend === 'metal') {
    const cores = [system.cpu_cores && `${system.cpu_cores} CPU`, system.gpu_cores && `${system.gpu_cores} GPU`].filter(Boolean).join(' / ');
    return [system.cpu_name || gpu, cores, ram].filter(Boolean).join(' · ') || tr('硬件信息未上报', 'Hardware details unavailable');
  }
  return [gpu, ram, system.cpu_name].filter(Boolean).join(' · ') || tr('硬件信息未上报', 'Hardware details unavailable');
}

function formatDate(value?: string | null): string {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleDateString();
}

function memoryBudgetPercentage(status: ModelConfigurationStatus): number | null {
  const required = status.required_memory_bytes;
  const budget = status.available_memory_bytes;
  if (required == null || budget == null || budget <= 0) return null;
  return (required / budget) * 100;
}

function memoryPressureColor(percentage: number): string {
  const hue = 120 - Math.min(Math.max(percentage, 0), 100) * 1.2;
  return `hsl(${hue.toFixed(1)} 68% 42%)`;
}

function configurationReasons(status: ModelConfigurationStatus, tr: Translate): string[] {
  return status.reasons.map((reason) => {
    if (reason === "Configuration requirements could not be determined.") return tr("无法确定此配置的资源需求。", reason);
    if (reason === "Estimated memory requirement exceeds the detected runtime budget.") return tr("预计内存需求超出检测到的推理预算。", reason);
    if (reason === "Estimated memory requirement exceeds the detected runtime budget, but at least 70% is available.") return tr("预计内存需求超出检测到的推理预算，但当前预算已达到需求的 70%。", reason);
    if (reason === "The detected runtime budget is below 70% of the estimated memory requirement.") return tr("检测到的推理预算不足预计内存需求的 70%。", reason);
    if (reason === "Fits within the detected runtime memory budget.") return tr("符合检测到的推理内存预算。", reason);
    if (reason === "Close other memory-heavy applications before loading.") return tr("加载前建议关闭其他占用大量内存的应用。", reason);
    if (reason === "All published precision tiers fit within the detected runtime memory budget.") return tr("所有已发布精度档都符合检测到的推理内存预算。", reason);
    if (reason === "More than half of the published precision tiers fit within the detected runtime memory budget.") return tr("超过一半的已发布精度档符合检测到的推理内存预算。", reason);
    if (reason === "At most half of the published precision tiers fit within the detected runtime memory budget.") return tr("不超过一半的已发布精度档符合检测到的推理内存预算。", reason);
    if (reason === "The detected runtime memory budget covers at least 70% of the smallest published precision tier.") return tr("检测到的推理预算已达到最小精度档内存需求的 70%，请谨慎加载。", reason);
    if (reason === "The detected runtime memory budget is below 70% of the smallest published precision tier.") return tr("检测到的推理预算不足最小精度档内存需求的 70%。", reason);
    if (reason === "All published precision tiers fit fully within the detected runtime memory budget.") return tr("所有已发布精度档都可完整常驻于当前推理内存预算。", reason);
    if (reason === "More than half of the published precision tiers fit fully within the detected runtime memory budget.") return tr("超过一半的已发布精度档可完整常驻于当前推理内存预算。", reason);
    if (reason === "At most half of the published precision tiers fit fully within the detected runtime memory budget.") return tr("不超过一半的已发布精度档可完整常驻于当前推理内存预算。", reason);
    if (reason === "The detected runtime memory budget covers at least 70% of the full-residency requirement for the smallest published precision tier.") return tr("当前推理预算已达到最小精度档完整常驻需求的 70%，处于临界区间。", reason);
    if (reason === "The detected runtime memory budget is below 70% of the full-residency requirement for the smallest published precision tier.") return tr("当前推理预算不足最小精度档完整常驻需求的 70%。", reason);
    if (reason === "The catalog is available offline; repository metadata could not be refreshed.") return tr("官方目录仍可离线浏览，但仓库元数据暂未刷新。", reason);
    if (reason === "GGUF must be converted before the MFQ runtime can load it.") return tr("MFQ Runtime 加载前需要先转换 GGUF。", reason);
    if (reason === "The repository can be downloaded, but direct runtime compatibility has not been verified.") return tr("可以下载此仓库，但尚未验证能否由 MFQ Runtime 直接加载。", reason);
    if (reason === "This format is not directly loadable by MFQ.") return tr("MFQ 无法直接加载此格式。", reason);
    if (reason === "This model architecture is not registered in MFQ.") return tr("MFQ 尚未注册此模型架构。", reason);
    if (reason === "Runtime compatibility could not be verified from repository metadata.") return tr("无法根据仓库元数据验证 MFQ Runtime 兼容性。", reason);
    return reason;
  });
}

function safeSegment(value: string): string {
  return value.replace(/[^A-Za-z0-9_.-]+/g, "-").replace(/^-+|-+$/g, "") || "model";
}

function downloadPatterns(variant: HubModelVariant | null): string[] {
  if (!variant) return [];
  if (variant.format === "unknown") return [];
  const weights = variant.files.length <= 48
    ? variant.files
    : variant.format === "mfq"
      ? [`${variant.label}*.mfq`]
      : variant.format === "hf"
        ? ["*.safetensors", "*.bin"]
        : variant.format === "gguf"
          ? ["*.gguf"]
          : variant.files.slice(0, 48);
  return Array.from(new Set([...weights, ...SUPPORT_FILES])).slice(0, 64);
}

function ConfigurationBadge({ status, tr }: { status: ModelConfigurationStatus; tr: Translate }) {
  const reason = configurationReasons(status, tr).join(" ");
  const presentation = {
    three_stars: ["★★★", tr("内存压力低：全部精度档均可完整常驻", "Low memory pressure: all precision tiers fit fully in memory")],
    two_stars: ["★★", tr("内存压力中等：多数精度档可完整常驻", "Moderate memory pressure: most precision tiers fit fully in memory")],
    one_star: ["★", tr("内存压力较高：仅部分精度档可完整常驻", "High memory pressure: only some precision tiers fit fully in memory")],
    caution: ["▲", tr("内存临界：最低档接近完整常驻门槛", "Memory near limit: the smallest tier is close to fitting fully")],
    not_recommended: ["✕", tr("内存不足：最低档无法完整常驻", "Insufficient memory: the smallest tier does not fit fully")],
    unknown: ["?", tr("内存压力未知", "Memory pressure unknown")],
  }[status.recommendation];
  return (
    <span
      aria-label={presentation[1]}
      className={`configuration-badge ${status.status} ${status.recommendation.replaceAll("_", "-")}`}
      title={[presentation[1], reason].filter(Boolean).join(" · ")}
    >
      <b aria-hidden="true">{presentation[0]}</b>
    </span>
  );
}

function ConfigurationDetails({ status, tr }: { status: ModelConfigurationStatus; tr: Translate }) {
  return (
    <div className={`configuration-details ${status.status} ${status.recommendation.replaceAll("_", "-")}`}>
      <div>
        <span>{tr("预计最低内存", "Estimated minimum memory")}</span>
        <strong>{formatBytes(status.required_memory_bytes)}</strong>
      </div>
      <div>
        <span>{tr("当前推理预算", "Detected runtime budget")}</span>
        <strong>{formatBytes(status.available_memory_bytes)}</strong>
      </div>
      <p>{configurationReasons(status, tr).join(" ")}</p>
    </div>
  );
}

function VariantList({
  disabled,
  onDownload,
  tr,
  variants,
}: {
  disabled: boolean;
  onDownload(variant: HubModelVariant, origin: DownloadOrigin): void;
  tr: Translate;
  variants: HubModelVariant[];
}) {
  if (!variants.length) {
    return <div className="model-browser-empty compact">{tr("仓库暂未返回可下载权重。", "No downloadable weights were returned by this repository.")}</div>;
  }
  return (
    <div className="model-variant-list">
      {variants.map((variant) => {
        const percentage = memoryBudgetPercentage(variant.configuration);
        const percentageLabel = percentage == null ? "—" : `${percentage.toFixed(1)}%`;
        const color = percentage == null ? "var(--muted)" : memoryPressureColor(percentage);
        return (
          <div className="model-variant" key={variant.id}>
            <div>
              <strong>{variant.label}</strong>
              <small>{variant.precision || variant.format.toUpperCase()} · {tr("文件", "file")} {formatBytes(variant.byte_size)} · {tr("完整常驻约", "est. full residency")} {formatBytes(variant.configuration.required_memory_bytes)}</small>
            </div>
            <div className="variant-memory-pressure">
              <div
                aria-label={`${tr("预计占当前推理预算", "Estimated share of runtime budget")}: ${percentageLabel}`}
                aria-valuemax={100}
                aria-valuemin={0}
                aria-valuenow={percentage == null ? undefined : Math.min(percentage, 100)}
                aria-valuetext={percentageLabel}
                className="variant-memory-track"
                role="progressbar"
              >
                <span style={{ backgroundColor: color, width: `${Math.min(percentage ?? 0, 100)}%` }} />
              </div>
              <strong style={{ color }}>{percentageLabel}</strong>
            </div>
            <button disabled={disabled} onClick={(event) => {
              const rect = event.currentTarget.getBoundingClientRect();
              onDownload(variant, { x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 });
            }} type="button">
              {tr("下载", "Download")}
            </button>
          </div>
        );
      })}
    </div>
  );
}

export function ModelBrowser({ jobKinds, onError, onJobCreated, tab, onTabChange, downloadQueue, tr }: ModelBrowserProps) {
  const [official, setOfficial] = useState<OfficialModelList | null>(null);
  const [officialSelection, setOfficialSelection] = useState<string | null>(null);
  const [officialSource, setOfficialSource] = useState<OfficialModelSource | null>(null);
  const [officialSourceInfo, setOfficialSourceInfo] = useState<HubModelInfo | null>(null);
  const [provider, setProvider] = useState<HubModelSummary["provider"]>("huggingface");
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<HubModelSummary[]>([]);
  const [communityModel, setCommunityModel] = useState<HubModelInfo | null>(null);
  const [officialLoading, setOfficialLoading] = useState(false);
  const [catalogLoading, setCatalogLoading] = useState(false);
  const [communityLoading, setCommunityLoading] = useState(false);
  const [downloading, setDownloading] = useState<string | null>(null);
  const officialRequest = useRef(0);
  const catalogRequest = useRef(0);
  const catalogTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const catalogController = useRef<AbortController | null>(null);
  const communityRequest = useRef(0);

  const selectedOfficial = useMemo(
    () => official?.data.find((item) => item.id === officialSelection) ?? official?.data[0] ?? null,
    [official, officialSelection],
  );
  const selectedSource = selectedOfficial?.sources.find((source) =>
    source.provider === officialSource?.provider && source.repo_id === officialSource.repo_id)
    ?? selectedOfficial?.selected_source ?? null;
  const sourceInfoMatches = !!officialSourceInfo && officialSourceInfo.provider === selectedSource?.provider
    && officialSourceInfo?.repo_id === selectedSource?.repo_id
    && (!selectedSource?.revision || officialSourceInfo?.revision === selectedSource.revision);
  const officialVariants = sourceInfoMatches ? officialSourceInfo!.variants
    : selectedSource?.provider === selectedOfficial?.selected_source.provider
      ? selectedOfficial?.variants ?? [] : [];
  const canDownload = (targetProvider: HubModelSummary["provider"]) =>
    jobKinds.some((item) => item.kind === `download.${targetProvider}`);

  async function loadOfficial(refresh = false) {
    const request = ++catalogRequest.current;
    if (catalogTimer.current !== null) clearTimeout(catalogTimer.current);
    catalogController.current?.abort();
    const controller = new AbortController();
    catalogController.current = controller;
    setCatalogLoading(true);
    async function readCatalog(force: boolean) {
      try {
        const catalog = await modelsApi.officialHubModels(force, controller.signal);
        if (request !== catalogRequest.current) return;
        setOfficial(catalog);
        setOfficialSelection((current) => current && catalog.data.some((item) => item.id === current)
          ? current : catalog.data[0]?.id ?? null);
        if (catalog.refreshing) {
          catalogTimer.current = setTimeout(() => void readCatalog(false), 1000);
        }
      } catch (cause) {
        if (request === catalogRequest.current && !controller.signal.aborted) {
          onError(cause instanceof Error ? cause.message : String(cause));
        }
      } finally {
        if (request === catalogRequest.current) setCatalogLoading(false);
      }
    }
    await readCatalog(refresh);
  }

  useEffect(() => {
    void loadOfficial(false);
    return () => {
      catalogRequest.current += 1;
      officialRequest.current += 1;
      communityRequest.current += 1;
      catalogController.current?.abort();
      if (catalogTimer.current !== null) clearTimeout(catalogTimer.current);
    };
  }, []);

  function chooseOfficial(item: OfficialModelInfo) {
    officialRequest.current += 1;
    setOfficialSelection(item.id);
    setOfficialSource(null);
    setOfficialSourceInfo(null);
    setOfficialLoading(false);
  }

  async function chooseOfficialSource(source: OfficialModelSource) {
    const request = ++officialRequest.current;
    setOfficialSource(source);
    setOfficialSourceInfo(null);
    if (!source.available) {
      setOfficialLoading(false);
      return;
    }
    setOfficialLoading(true);
    try {
      const info = await modelsApi.hubModelInfo(
        source.provider,
        source.repo_id,
        source.revision || undefined,
      );
      if (request === officialRequest.current) setOfficialSourceInfo(info);
    } catch (cause) {
      if (request === officialRequest.current) {
        setOfficialSource(null);
        onError(cause instanceof Error ? cause.message : String(cause));
      }
    } finally {
      if (request === officialRequest.current) setOfficialLoading(false);
    }
  }

  async function search(event: FormEvent) {
    event.preventDefault();
    const reference = query.trim();
    if (!reference) return;
    const request = ++communityRequest.current;
    setCommunityLoading(true);
    try {
      if (reference.includes("/") || /^https?:\/\//i.test(reference)) {
        const model = await modelsApi.resolveHubModel(reference, provider);
        if (request !== communityRequest.current) return;
        setProvider(model.provider);
        setCommunityModel(model);
        setResults([model]);
      } else {
        const found = await modelsApi.searchHubModels(provider, reference);
        if (request !== communityRequest.current) return;
        setResults(found);
        setCommunityModel(null);
      }
    } catch (cause) {
      if (request === communityRequest.current) onError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      if (request === communityRequest.current) setCommunityLoading(false);
    }
  }

  async function inspect(item: HubModelSummary) {
    const request = ++communityRequest.current;
    setCommunityLoading(true);
    try {
      const model = await modelsApi.hubModelInfo(item.provider, item.repo_id);
      if (request === communityRequest.current) setCommunityModel(model);
    } catch (cause) {
      if (request === communityRequest.current) onError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      if (request === communityRequest.current) setCommunityLoading(false);
    }
  }

  async function download(
    source: { provider: HubModelSummary["provider"]; repo_id: string; revision: string },
    variant: HubModelVariant,
    origin: DownloadOrigin,
  ) {
    const marker = `${source.provider}:${source.repo_id}:${variant.id}`;
    setDownloading(marker);
    try {
      const repositoryPath = source.repo_id.split("/").map(safeSegment).join("/");
      const variantPath = safeSegment(variant.label);
      const created = await jobsApi.createJob(`download.${source.provider}`, {
        repo_id: source.repo_id,
        destination: `models/${source.provider}/${repositoryPath}/${variantPath}`,
        revision: source.revision,
        include: downloadPatterns(variant),
        expected_bytes: variant.byte_size || null,
      });
      onJobCreated(created, origin);
    } catch (cause) {
      onError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setDownloading(null);
    }
  }

  const system = official?.system;
  return (
    <section className="model-browser">
      <header className="model-browser-header">
        <div>
          <h2>{tr("模型浏览器", "Model browser")}</h2>
          <p>{tr("查找官方优化模型或浏览社区仓库。", "Find optimized official models or browse community repositories.")}</p>
        </div>
        <div className="model-browser-tabs" role="tablist">
          <button aria-selected={tab === "official"} onClick={() => onTabChange("official")} role="tab" type="button">{tr("官方模型", "Official")}</button>
          <button aria-selected={tab === "community"} onClick={() => onTabChange("community")} role="tab" type="button">{tr("第三方模型", "Community")}</button>
          <button aria-selected={tab === "downloads"} onClick={() => onTabChange("downloads")} role="tab" type="button">{tr("下载队列", "Download queue")}</button>
        </div>
      </header>

      {tab === "official" ? (
        <>
          <div className="detected-configuration">
            <div><span>{tr("检测到的配置", "Detected configuration")}</span><div className="detected-hardware-summary"><strong>{system ? hardwareSummary(system, tr) : tr("正在检测", "Detecting")}</strong>{system && <BackendBadge backend={system.backend} />}</div></div>
            <div><span>{tr("推理预算", "Runtime budget")}</span><strong>{formatBytes(system?.runtime_memory_budget_bytes)}</strong></div>
            <button disabled={catalogLoading || official?.refreshing} onClick={() => void loadOfficial(true)} type="button">{catalogLoading || official?.refreshing ? tr("刷新中", "Refreshing") : tr("刷新目录", "Refresh")}</button>
          </div>
          <div className="memory-pressure-guide">
            <strong>{tr("内存压力", "Memory pressure")}</strong>
            <span><b>★★★</b>{tr("低", "Low")}</span>
            <span><b>★★</b>{tr("中", "Moderate")}</span>
            <span><b>★</b>{tr("高", "High")}</span>
            <span><b>▲</b>{tr("临界", "Near limit")}</span>
            <span><b>✕</b>{tr("不足", "Insufficient")}</span>
            <small>{tr("只反映当前设备可完整常驻的精度档比例，不代表模型能力或质量。", "Reflects only how many precision tiers fit fully in this device's memory, not model capability or quality.")}</small>
          </div>
          <div className="model-browser-layout">
            <div className="official-model-grid">
              {official?.data.map((item) => (
                <button className={selectedOfficial?.id === item.id ? "official-model-card selected" : "official-model-card"} key={item.id} onClick={() => chooseOfficial(item)} type="button">
                  <div className="official-model-card-title"><span>{item.name.slice(0, 1).toUpperCase()}</span><div><strong>{item.name}</strong><small>{item.family}</small></div></div>
                  <p>{tr(item.description_zh, item.description)}</p>
                  <div className="model-chip-row">{item.precision_options.map((value) => <span key={value}>{value}</span>)}</div>
                  <ConfigurationBadge status={item.configuration} tr={tr} />
                </button>
              ))}
              {!official && <div className="model-browser-empty">{tr("正在读取官方目录…", "Loading the official catalog…")}</div>}
            </div>
            {selectedOfficial && selectedSource && (
              <aside className="model-detail-panel">
                <div className="model-detail-heading"><div><small>{selectedOfficial.family}</small><h3>{selectedOfficial.name}</h3></div><ConfigurationBadge status={selectedOfficial.configuration} tr={tr} /></div>
                <p>{tr(selectedOfficial.description_zh, selectedOfficial.description)}</p>
                <ConfigurationDetails status={selectedOfficial.configuration} tr={tr} />
                <dl className="model-metadata-grid">
                  <div><dt>{tr("架构", "Architecture")}</dt><dd>{selectedOfficial.architecture}</dd></div>
                  <div><dt>{tr("参数", "Parameters")}</dt><dd>{selectedOfficial.parameter_label || "—"}</dd></div>
                  <div><dt>{tr("激活参数", "Active parameters")}</dt><dd>{selectedOfficial.active_parameter_label || "—"}</dd></div>
                  <div><dt>{tr("模态", "Modalities")}</dt><dd>{selectedOfficial.modalities.join(" · ")}</dd></div>
                  <div><dt>{tr("功能", "Capabilities")}</dt><dd>{selectedOfficial.capabilities.join(" · ")}</dd></div>
                  <div><dt>{tr("许可", "License")}</dt><dd>{selectedOfficial.license || tr("查看模型卡", "See model card")}</dd></div>
                  <div><dt>{tr("下载", "Downloads")}</dt><dd>{formatCount(selectedOfficial.downloads)}</dd></div>
                  <div><dt>{tr("收藏", "Likes")}</dt><dd>{formatCount(selectedOfficial.likes)}</dd></div>
                  <div><dt>{tr("更新时间", "Updated")}</dt><dd>{formatDate(selectedOfficial.updated_at)}</dd></div>
                </dl>
                <label className="model-source-picker"><span>{tr("下载来源", "Download source")}</span><select onChange={(event) => { const source = selectedOfficial.sources[Number(event.target.value)]; if (source) void chooseOfficialSource(source); }} value={String(Math.max(0, selectedOfficial.sources.findIndex((item) => item.provider === selectedSource.provider && item.repo_id === selectedSource.repo_id)))}>{selectedOfficial.sources.map((source, index) => <option disabled={!source.available} key={`${source.provider}:${source.repo_id}`} value={index}>{source.provider === "huggingface" ? "Hugging Face" : "ModelScope"}{source.available ? "" : ` · ${tr("离线", "unavailable")}`}</option>)}</select></label>
                <div className="repository-line"><button onClick={() => void openStudioExternal(selectedSource.url).catch((cause) => onError(cause instanceof Error ? cause.message : String(cause)))} type="button">{selectedSource.repo_id}</button><span>{officialVariants.length} {tr("个精度版本", "variants")}</span></div>
                {selectedOfficial.supports_ssd_streaming && <div className="streaming-note">{tr("支持 SSD 专家流式读取；即使无法完整常驻仍可流式运行。上方图标只表示完整常驻时的内存压力。", "SSD expert streaming remains available when the model cannot fit fully in memory. The icon above reflects full-residency memory pressure only.")}</div>}
                <VariantList disabled={officialLoading || !selectedSource.available || !canDownload(selectedSource.provider) || downloading !== null} onDownload={(variant, origin) => void download({ provider: selectedSource.provider, repo_id: selectedSource.repo_id, revision: selectedSource.revision || selectedOfficial.revision }, variant, origin)} tr={tr} variants={officialVariants} />
              </aside>
            )}
          </div>
        </>
      ) : tab === 'community' ? (
        <>
          <form className="community-model-search" onSubmit={search}>
            <select onChange={(event) => setProvider(event.target.value as HubModelSummary["provider"])} value={provider}><option value="huggingface">Hugging Face</option><option value="modelscope">ModelScope</option></select>
            <input onChange={(event) => setQuery(event.target.value)} placeholder={tr("模型名称、owner/repo 或仓库链接", "Model name, owner/repo, or repository URL")} value={query} />
            <button disabled={communityLoading || !query.trim()} type="submit">{communityLoading ? tr("查找中", "Searching") : tr("查找", "Search")}</button>
          </form>
          <p className="community-search-hint">{tr("支持直接粘贴 Hugging Face 与 ModelScope 链接；下载仍由可续传后台任务管理。", "Paste a Hugging Face or ModelScope link directly; downloads remain resumable background jobs.")}</p>
          <div className="model-browser-layout community-layout">
            <div className="community-results">
              {results.map((item) => <button className={communityModel?.repo_id === item.repo_id && communityModel.provider === item.provider ? "selected" : ""} key={`${item.provider}:${item.repo_id}`} onClick={() => void inspect(item)} type="button"><div><strong>{item.repo_id}</strong><small>{item.provider === "huggingface" ? "Hugging Face" : "ModelScope"} · {formatCount(item.downloads)} downloads</small></div><span>{formatBytes(item.total_bytes)}</span></button>)}
              {!results.length && <div className="model-browser-empty">{tr("搜索社区模型，或粘贴仓库链接直接打开。", "Search community models or paste a repository link to open it directly.")}</div>}
            </div>
            {communityModel && (
              <aside className="model-detail-panel">
                <div className="model-detail-heading"><div><small>{communityModel.author || communityModel.provider}</small><h3>{communityModel.repo_id.split("/").pop()}</h3></div>{communityModel.gated && <span className="gated-badge">{tr("需要授权", "Gated")}</span>}</div>
                {communityModel.description && <p>{communityModel.description}</p>}
                <dl className="model-metadata-grid">
                  <div><dt>{tr("架构", "Architecture")}</dt><dd>{communityModel.architectures.join(", ") || tr("未声明", "Not declared")}</dd></div>
                  <div><dt>{tr("参数", "Parameters")}</dt><dd>{communityModel.parameter_count ? formatCount(communityModel.parameter_count) : "—"}</dd></div>
                  <div><dt>{tr("模态", "Modalities")}</dt><dd>{communityModel.modalities.join(" · ") || "text"}</dd></div>
                  <div><dt>{tr("许可", "License")}</dt><dd>{communityModel.license || tr("未声明", "Not declared")}</dd></div>
                  <div><dt>{tr("仓库大小", "Repository size")}</dt><dd>{formatBytes(communityModel.total_bytes)}</dd></div>
                  <div><dt>{tr("版本", "Revision")}</dt><dd title={communityModel.revision}>{communityModel.revision.slice(0, 12)}</dd></div>
                  <div><dt>{tr("框架", "Library")}</dt><dd>{communityModel.library || "—"}</dd></div>
                  <div><dt>{tr("任务", "Task")}</dt><dd>{communityModel.pipeline_tag || "—"}</dd></div>
                  <div><dt>{tr("更新时间", "Updated")}</dt><dd>{formatDate(communityModel.updated_at)}</dd></div>
                  <div><dt>{tr("MFQ 兼容性", "MFQ compatibility")}</dt><dd>{communityModel.runtime_compatible === true ? tr("已验证", "Verified") : communityModel.runtime_compatible === false ? tr("暂不支持", "Unsupported") : tr("未知", "Unknown")}</dd></div>
                </dl>
                {communityModel.source_url && <div className="repository-line"><button onClick={() => void openStudioExternal(communityModel.source_url!).catch((cause) => onError(cause instanceof Error ? cause.message : String(cause)))} type="button">{tr("打开模型卡", "Open model card")}</button><span>{formatCount(communityModel.downloads)} downloads · {formatCount(communityModel.likes)} likes · {communityModel.files.length} files</span></div>}
                <VariantList disabled={communityLoading || !canDownload(communityModel.provider) || downloading !== null} onDownload={(variant, origin) => void download({ provider: communityModel.provider, repo_id: communityModel.repo_id, revision: communityModel.revision }, variant, origin)} tr={tr} variants={communityModel.variants} />
              </aside>
            )}
          </div>
        </>
      ) : downloadQueue}
    </section>
  );
}
