import { FormEvent, useEffect, useMemo, useRef, useState } from "react";

import type {
  HubModelInfo,
  HubModelSummary,
  HubModelVariant,
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

type Translate = (chinese: string, english: string) => string;

interface ModelBrowserProps {
  jobKinds: JobKindResource[];
  onError(message: string): void;
  onJobCreated(job: JobResource): void;
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

function formatDate(value?: string | null): string {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleDateString();
}

function configurationReasons(status: ModelConfigurationStatus, tr: Translate): string[] {
  return status.reasons.map((reason) => {
    if (reason === "Configuration requirements could not be determined.") return tr("无法确定此配置的资源需求。", reason);
    if (reason === "Estimated memory requirement exceeds the detected runtime budget.") return tr("预计内存需求超出检测到的推理预算。", reason);
    if (reason === "Fits within the detected runtime memory budget.") return tr("符合检测到的推理内存预算。", reason);
    if (reason === "Close other memory-heavy applications before loading.") return tr("加载前建议关闭其他占用大量内存的应用。", reason);
    if (reason === "At least one published precision tier fits this configuration.") return tr("至少有一个已发布精度档适合当前配置。", reason);
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
  const recommended = status.status === "recommended";
  const warning = status.status === "warning";
  const title = configurationReasons(status, tr).join(" ");
  return (
    <span
      className={`configuration-badge ${status.status}`}
      title={title}
    >
      <b aria-hidden="true">{recommended ? "★" : warning ? "⚠" : "?"}</b>
      {recommended
        ? tr("适合此设备", "Recommended")
        : warning
          ? tr("超出建议配置", "Check requirements")
          : tr("配置未知", "Unknown")}
    </span>
  );
}

function ConfigurationDetails({ status, tr }: { status: ModelConfigurationStatus; tr: Translate }) {
  return (
    <div className={`configuration-details ${status.status}`}>
      <div>
        <span>{tr("预计最低内存", "Estimated minimum")}</span>
        <strong>{formatBytes(status.required_memory_bytes)}</strong>
      </div>
      <div>
        <span>{tr("建议内存", "Recommended memory")}</span>
        <strong>{formatBytes(status.recommended_memory_bytes)}</strong>
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
  onDownload(variant: HubModelVariant): void;
  tr: Translate;
  variants: HubModelVariant[];
}) {
  if (!variants.length) {
    return <div className="model-browser-empty compact">{tr("仓库暂未返回可下载权重。", "No downloadable weights were returned by this repository.")}</div>;
  }
  return (
    <div className="model-variant-list">
      {variants.map((variant) => (
        <div className="model-variant" key={variant.id}>
          <div>
            <strong>{variant.label}</strong>
            <small>{variant.precision || variant.format.toUpperCase()} · {formatBytes(variant.byte_size)} · {tr("预计", "est.")} {formatBytes(variant.configuration.required_memory_bytes)} RAM</small>
          </div>
          <ConfigurationBadge status={variant.configuration} tr={tr} />
          <button disabled={disabled} onClick={() => onDownload(variant)} type="button">
            {tr("下载", "Download")}
          </button>
        </div>
      ))}
    </div>
  );
}

export function ModelBrowser({ jobKinds, onError, onJobCreated, tr }: ModelBrowserProps) {
  const [tab, setTab] = useState<"official" | "community">("official");
  const [official, setOfficial] = useState<OfficialModelList | null>(null);
  const [officialSelection, setOfficialSelection] = useState<string | null>(null);
  const [officialSource, setOfficialSource] = useState<OfficialModelSource | null>(null);
  const [officialSourceInfo, setOfficialSourceInfo] = useState<HubModelInfo | null>(null);
  const [provider, setProvider] = useState<HubModelSummary["provider"]>("huggingface");
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<HubModelSummary[]>([]);
  const [communityModel, setCommunityModel] = useState<HubModelInfo | null>(null);
  const [officialLoading, setOfficialLoading] = useState(false);
  const [communityLoading, setCommunityLoading] = useState(false);
  const [downloading, setDownloading] = useState<string | null>(null);
  const officialRequest = useRef(0);
  const communityRequest = useRef(0);

  const selectedOfficial = useMemo(
    () => official?.data.find((item) => item.id === officialSelection) ?? official?.data[0] ?? null,
    [official, officialSelection],
  );
  const selectedSource = officialSource ?? selectedOfficial?.selected_source ?? null;
  const officialVariants = officialSourceInfo?.variants ?? selectedOfficial?.variants ?? [];
  const canDownload = (targetProvider: HubModelSummary["provider"]) =>
    jobKinds.some((item) => item.kind === `download.${targetProvider}`);

  async function loadOfficial(refresh = false) {
    const request = ++officialRequest.current;
    setOfficialLoading(true);
    try {
      const catalog = await modelsApi.officialHubModels(refresh);
      if (request !== officialRequest.current) return;
      setOfficial(catalog);
      setOfficialSelection((current) => current && catalog.data.some((item) => item.id === current)
        ? current
        : catalog.data[0]?.id ?? null);
      setOfficialSource(null);
      setOfficialSourceInfo(null);
    } catch (cause) {
      if (request === officialRequest.current) onError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      if (request === officialRequest.current) setOfficialLoading(false);
    }
  }

  useEffect(() => {
    void loadOfficial(false);
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
      onJobCreated(created);
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
          <button aria-selected={tab === "official"} onClick={() => setTab("official")} role="tab" type="button">{tr("官方模型", "Official")}</button>
          <button aria-selected={tab === "community"} onClick={() => setTab("community")} role="tab" type="button">{tr("第三方模型", "Community")}</button>
        </div>
      </header>

      {tab === "official" ? (
        <>
          <div className="detected-configuration">
            <div><span>{tr("检测到的配置", "Detected configuration")}</span><strong>{system ? `${system.platform} · ${system.machine} · ${system.backend.toUpperCase()}` : tr("正在检测", "Detecting")}</strong></div>
            <div><span>{tr("物理内存", "Physical memory")}</span><strong>{formatBytes(system?.physical_memory_bytes)}</strong></div>
            <div><span>{tr("推理预算", "Runtime budget")}</span><strong>{formatBytes(system?.runtime_memory_budget_bytes)}</strong></div>
            <button disabled={officialLoading} onClick={() => void loadOfficial(true)} type="button">{officialLoading ? tr("刷新中", "Refreshing") : tr("刷新目录", "Refresh")}</button>
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
                {selectedOfficial.supports_ssd_streaming && <div className="streaming-note">{tr("支持 SSD 专家流式读取；星级按建议常驻预算评估，不要求整个模型进入内存。", "SSD expert streaming is supported; the rating uses the recommended resident budget rather than requiring the full model in memory.")}</div>}
                <VariantList disabled={officialLoading || !selectedSource.available || !canDownload(selectedSource.provider) || downloading !== null} onDownload={(variant) => void download({ provider: selectedSource.provider, repo_id: selectedSource.repo_id, revision: selectedSource.revision || selectedOfficial.revision }, variant)} tr={tr} variants={officialVariants} />
              </aside>
            )}
          </div>
        </>
      ) : (
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
                <VariantList disabled={communityLoading || !canDownload(communityModel.provider) || downloading !== null} onDownload={(variant) => void download({ provider: communityModel.provider, repo_id: communityModel.repo_id, revision: communityModel.revision }, variant)} tr={tr} variants={communityModel.variants} />
              </aside>
            )}
          </div>
        </>
      )}
    </section>
  );
}
