import { FormEvent, type ReactNode, useEffect, useMemo, useRef, useState } from "react";

import type {
  HubModelInfo,
  HubMemoryPool,
  HubModelSummary,
  HubModelVariant,
  HubSystemProfile,
  JobKindResource,
  JobResource,
  ModelConfigurationStatus,
  ModelParameterBreakdown,
  OfficialModelInfo,
  OfficialModelList,
  OfficialModelSource,
} from '../../shared/api/types';
import { jobsApi } from '../../shared/api/resources/jobs';
import { modelsApi } from '../../shared/api/resources/models';
import { openStudioExternal } from '../../shared/platform/studio';
import { BackendBadge } from './BackendBadge';
import { ModelVendorMark } from '../../app/ModelVendorMark';
import { ModelPressureBars } from './ModelMemoryPressure';
import { variantMemoryPressureRows, type PressureRow } from './memoryPressure';
import { RepositoryFiles } from './RepositoryFiles';
import { estimateCacheBytes, KvCachePlanner, plannedConfiguration, withCacheMemory } from './KvCachePlanner';

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

function formatParameters(value: number): string {
  if (value >= 1e9) return `${(value / 1e9).toFixed(1).replace(/\.0$/, '')}B`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(1).replace(/\.0$/, '')}M`;
  return new Intl.NumberFormat().format(value);
}

function modalityLabels(values: string[], tr: Translate): string {
  const labels: Record<string, string> = {
    text: tr('文本', 'text'), image: tr('图像', 'image'),
    video: tr('视频', 'video'), audio: tr('音频', 'audio'),
  };
  return values.map((value) => labels[value] || value).join(' · ');
}

function ParameterValue({ parameters, label, tr }: { parameters?: ModelParameterBreakdown | null; label?: string | null; tr: Translate }) {
  if (!parameters) return <>{label || '—'}</>;
  return <>{formatParameters(parameters.total)}{tr('（', ' (')}{formatParameters(parameters.dense)} {tr('稠密', 'dense')}
    {parameters.routed_experts > 0 && <> + {formatParameters(parameters.routed_experts)} {tr('路由专家', 'routed experts')}</>}
    {parameters.ple > 0 && <> + {formatParameters(parameters.ple)} PLE</>}{tr('）', ')')}</>;
}

function MtpSupport({ supported, tr }: { supported?: boolean | null; tr: Translate }) {
  return <div className="model-mtp-support">
    <dt>{tr('MTP支持', 'MTP support')}</dt>
    <dd>{supported == null ? tr('未确认', 'Unknown') : supported ? tr('是', 'Yes') : tr('否', 'No')}</dd>
  </div>;
}

function hardwareSummary(system: HubSystemProfile, tr: Translate): string {
  const unified = system.memory_pools?.some((pool) => pool.kind === 'uma') || /^Apple M\d+(?: (?:Pro|Max|Ultra))?$/.test(system.cpu_name || '');
  const ram = system.physical_memory_bytes ? `${formatBytes(system.physical_memory_bytes).replace(/\.0 /, ' ')} ${unified ? 'URAM' : 'RAM'}` : null;
  const gpu = system.gpu_names?.join(' + ');
  if (system.backend === 'metal') {
    const cores = [system.cpu_cores && `${system.cpu_cores} CPU`, system.gpu_cores && `${system.gpu_cores} GPU`].filter(Boolean).join(' / ');
    return [system.cpu_name || gpu, cores, ram].filter(Boolean).join(' · ') || tr('硬件信息未上报', 'Hardware details unavailable');
  }
  return [gpu, ram, system.cpu_name].filter(Boolean).join(' · ') || tr('硬件信息未上报', 'Hardware details unavailable');
}

function MemoryBudget({ system, tr }: { system: HubSystemProfile | undefined; tr: Translate }) {
  const pools: HubMemoryPool[] = system?.memory_pools?.length ? system.memory_pools : system ? [{
    kind: /^Apple M\d+(?: (?:Pro|Max|Ultra))?$/.test(system.cpu_name || '') ? 'uma' : 'ram',
    capacity_bytes: system.physical_memory_bytes,
  }] : [];
  return <div className="detected-memory-pools">
    {pools.length ? pools.map((pool, index) => <div className="detected-memory-pool" key={`${pool.kind}:${index}`} title={pool.device || undefined}>
      <strong>{pool.capacity_bytes ? `${(pool.capacity_bytes / 2 ** 30).toFixed(1).replace(/\.0$/, '')} GiB` : '—'} {pool.kind === 'uma' ? 'URAM' : pool.kind.toUpperCase()}</strong>
      <small title={pool.bandwidth_bytes_per_second ? tr('规格带宽，非实测吞吐', 'Specified bandwidth, not measured throughput') : undefined}>
        {pool.bandwidth_bytes_per_second ? `${new Intl.NumberFormat(undefined, { maximumFractionDigits: 1 }).format(pool.bandwidth_bytes_per_second / 2 ** 30)} GiB/s` : tr('带宽未上报', 'Bandwidth unavailable')}
      </small>
    </div>) : <strong>—</strong>}
  </div>;
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
    if (reason === "Tensor payload baseline; excludes streamed PLE, KV cache and runtime repacking.") return tr("按张量载荷计算权重常驻基线，已扣除流式 PLE；不包含 KV 缓存和运行时重排开销。", reason);
    if (reason === "Estimated resident weights include a 10% allowance on packed or unclassified weights; exclude streamed PLE, KV, prefix caches and temporary workspaces.") return tr("量化或未知布局权重暂加 10% 存储余量，已扣除流式 PLE；KV、前缀缓存和临时工作区另计。", reason);
    if (reason === "File-size estimate with a 10% allowance; tensor metadata is unavailable.") return tr("暂按文件大小加 10% 余量估计，尚未取得张量元数据，未扣除 PLE。", reason);
    if (reason === "All published precision tiers' estimated resident weights fit within the detected runtime memory budget.") return tr("所有已发布精度档的预计权重常驻均在当前推理预算以内。", reason);
    if (reason === "More than half of the published precision tiers' estimated resident weights fit within the detected runtime memory budget.") return tr("超过一半的已发布精度档预计权重常驻在当前推理预算以内。", reason);
    if (reason === "At most half of the published precision tiers' estimated resident weights fit within the detected runtime memory budget.") return tr("仅部分已发布精度档的预计权重常驻在当前推理预算以内。", reason);
    if (reason === "The detected runtime memory budget covers at least 70% of the estimated resident weight requirement for the smallest published precision tier.") return tr("当前推理预算达到最低档预计权重常驻的 70%，处于临界区间。", reason);
    if (reason === "The detected runtime memory budget is below 70% of the estimated resident weight requirement for the smallest published precision tier.") return tr("当前推理预算不足最低档预计权重常驻的 70%。", reason);
    if (reason === "File-size estimate; tensor metadata is unavailable.") return tr("未读取到张量元数据，暂按文件大小估计权重占用。", reason);
    if (reason === "All published precision tiers' weight baselines fit within the detected runtime memory budget.") return tr("所有已发布精度档的权重基线均在当前推理预算以内。", reason);
    if (reason === "More than half of the published precision tiers' weight baselines fit within the detected runtime memory budget.") return tr("超过一半的已发布精度档权重基线在当前推理预算以内。", reason);
    if (reason === "At most half of the published precision tiers' weight baselines fit within the detected runtime memory budget.") return tr("仅部分已发布精度档的权重基线在当前推理预算以内。", reason);
    if (reason === "The detected runtime memory budget covers at least 70% of the weight baseline requirement for the smallest published precision tier.") return tr("当前推理预算达到最低档权重基线的 70%，处于临界区间。", reason);
    if (reason === "The detected runtime memory budget is below 70% of the weight baseline requirement for the smallest published precision tier.") return tr("当前推理预算不足最低档权重基线的 70%。", reason);
    return reason;
  });
}

function safeSegment(value: string): string {
  return value.replace(/[^A-Za-z0-9_.-]+/g, "-").replace(/^-+|-+$/g, "") || "model";
}

function downloadPatterns(variant: HubModelVariant | null): string[] {
  if (!variant) return [];
  if (variant.format === "unknown") return variant.files;
  const weights = variant.files.length <= 48
    ? variant.files
    : variant.format === "mfq"
      ? [`${variant.label}*.mfq`]
      : variant.format === "hf"
        ? ["*.safetensors", "*.bin"]
        : variant.format === "gguf"
          ? ["*.gguf"]
          : variant.files.slice(0, 48);
  return Array.from(new Set([...weights, ...SUPPORT_FILES]));
}

function ConfigurationDetails({ status, planned = false, rows, tr }: { status: ModelConfigurationStatus; planned?: boolean; rows?: PressureRow[]; tr: Translate }) {
  if (rows && rows.some(row => row.kind !== 'shared')) {
    return <div className="configuration-details">{rows.map(row => <div key={row.kind}>
      <span>{row.kind === 'vram' ? tr('最低档预计显存常驻 / 显存容量', 'Smallest tier VRAM residency / capacity')
        : tr('最低档预计 RAM 常驻 / RAM 容量', 'Smallest tier RAM residency / capacity')}</span>
      <strong>{formatBytes(row.required)} / {formatBytes(row.available)}</strong>
    </div>)}<p>{rows.length > 1
      ? tr('显存按稠密权重和已应用的 KV 计算；RAM 按路由专家和 Embedding 计算。SSD PLE 单独计算。', 'VRAM includes dense weights and applied KV; RAM includes routed experts and embedding. SSD PLE is separate.')
      : tr('显存按整模权重和已应用的 KV 计算，已扣除 SSD PLE。', 'VRAM includes model weights and applied KV, excluding SSD PLE.')}</p></div>;
  }
  return (
    <div className={`configuration-details ${status.status} ${status.recommendation.replaceAll("_", "-")}`}>
      <div>
        <span>{planned ? tr('最低档权重 + KV/递推状态', 'Smallest tier weights + KV/recurrent state') : tr("最低档预计权重常驻", "Smallest tier estimated resident weights")}</span>
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
  cacheBytes = 0,
  system,
  moe = false,
}: {
  disabled: boolean;
  onDownload(variant: HubModelVariant, origin: DownloadOrigin): void;
  tr: Translate;
  variants: HubModelVariant[];
  cacheBytes?: number;
  system?: HubSystemProfile;
  moe?: boolean;
}) {
  if (!variants.length) {
    return <div className="model-browser-empty compact">{tr("仓库暂未返回可下载权重。", "No downloadable weights were returned by this repository.")}</div>;
  }
  return (
    <div className="model-variant-list">
      {variants.map((variant) => {
        return (
          <div className="model-variant" key={variant.id}>
            <div>
              <strong>{variant.label}</strong>
              <small>{variant.precision || variant.format.toUpperCase()} · {tr("文件", "file")} {formatBytes(variant.byte_size)} · {variant.estimated_resident_weight_bytes != null ? tr("预计权重常驻", "est. resident weights") : variant.resident_weight_bytes != null ? tr("权重常驻基线", "resident weight baseline") : tr("权重占用估计", "est. weight memory")} {formatBytes(variant.estimated_resident_weight_bytes ?? variant.resident_weight_bytes ?? (variant.configuration.required_memory_bytes == null ? null : variant.configuration.required_memory_bytes - cacheBytes))}{(variant.ssd_ple_bytes ?? 0) > 0 && ` · SSD PLE ${formatBytes(variant.ssd_ple_bytes)}`}</small>
              {cacheBytes > 0 && <small>{tr('规划 KV/递推状态', 'Planned KV/recurrent state')} {formatBytes(cacheBytes)} · {tr('预计总常驻', 'Est. total residency')} {formatBytes(variant.configuration.required_memory_bytes)}</small>}
            </div>
            <ModelPressureBars weights={variant.estimated_resident_weight_bytes ?? variant.resident_weight_bytes
                ?? (variant.configuration.required_memory_bytes == null ? null : variant.configuration.required_memory_bytes - cacheBytes)}
              roles={variant.estimated_weight_bytes_by_role} system={system} cacheBytes={cacheBytes} moe={moe}
              available={variant.configuration.available_memory_bytes}
              label={tr("预计占当前推理预算", "Estimated share of runtime budget")}
              tr={tr} />
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
  const [maxWorkers, setMaxWorkers] = useState(8);
  const [plannedContexts, setPlannedContexts] = useState<Record<string, number | undefined>>({});
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
  const selectedMetadata = sourceInfoMatches ? officialSourceInfo : selectedSource?.provider === selectedOfficial?.selected_source.provider
    && selectedSource?.repo_id === selectedOfficial?.selected_source.repo_id ? selectedOfficial : null;
  const parameters = selectedMetadata?.parameter_breakdown;
  const mtpSupported = selectedMetadata?.mtp_supported;
  const officialPlanKey = `official:${selectedOfficial?.id}:${selectedSource?.provider}:${selectedSource?.repo_id}`;
  const officialContext = plannedContexts[officialPlanKey];
  const officialCacheBytes = selectedMetadata?.cache_profile && officialContext != null
    ? estimateCacheBytes(selectedMetadata.cache_profile, officialContext) : 0;
  const plannedOfficialVariants = withCacheMemory(officialVariants, officialCacheBytes);
  const officialConfiguration = selectedOfficial && officialCacheBytes > 0
    ? plannedConfiguration(plannedOfficialVariants, selectedOfficial.configuration, tr) : selectedOfficial?.configuration;
  const communityPlanKey = `community:${communityModel?.provider}:${communityModel?.repo_id}`;
  const communityContext = plannedContexts[communityPlanKey];
  const communityCacheBytes = communityModel?.cache_profile && communityContext != null
    ? estimateCacheBytes(communityModel.cache_profile, communityContext) : 0;
  const applyPlan = (key: string, context: number | undefined) => setPlannedContexts((current) => ({ ...current, [key]: context }));
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
        max_workers: maxWorkers,
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
        <div className="model-browser-actions">
          <label className="download-concurrency">{tr('最大并发下载上限', 'Maximum concurrent downloads')}
            <select aria-label={tr('最大并发下载上限', 'Maximum concurrent downloads')} value={maxWorkers} onChange={(event) => setMaxWorkers(Number(event.target.value))}>
              {[1, 2, 4, 8, 16].map((count) => <option key={count} value={count}>{count}</option>)}
            </select>
          </label>
        <div className="model-browser-tabs" role="tablist">
          <button aria-selected={tab === "official"} onClick={() => onTabChange("official")} role="tab" type="button">{tr("官方模型", "Official")}</button>
          <button aria-selected={tab === "community"} onClick={() => onTabChange("community")} role="tab" type="button">{tr("第三方模型", "Community")}</button>
          <button aria-selected={tab === "downloads"} onClick={() => onTabChange("downloads")} role="tab" type="button">{tr("下载队列", "Download queue")}</button>
        </div>
        </div>
      </header>

      {tab === "official" ? (
        <>
          <div className="detected-configuration">
            <div><span>{tr("检测到的配置", "Detected configuration")}</span><div className="detected-hardware-summary"><strong>{system ? hardwareSummary(system, tr) : tr("正在检测", "Detecting")}</strong>{system && <BackendBadge backend={system.backend} />}</div></div>
            <div><span>{tr("推理预算", "Runtime budget")}</span><MemoryBudget system={system || undefined} tr={tr} /></div>
            <button disabled={catalogLoading || official?.refreshing} onClick={() => void loadOfficial(true)} type="button">{catalogLoading || official?.refreshing ? tr("刷新中", "Refreshing") : tr("刷新目录", "Refresh")}</button>
          </div>
          <div className="model-browser-layout">
            <div className="official-model-grid">
              {official?.data.map((item) => (
                <button className={selectedOfficial?.id === item.id ? "official-model-card selected" : "official-model-card"} key={item.id} onClick={() => chooseOfficial(item)} type="button">
                  <div className="official-model-card-title"><span>{item.name.slice(0, 1).toUpperCase()}</span><div><strong>{item.name}</strong><small>{item.family}</small></div><ModelVendorMark name={item.name} architecture={item.architecture} size={28} /></div>
                  <p>{tr(item.description_zh, item.description)}</p>
                  <div className="model-chip-row">{item.precision_options.map((value) => <span key={value}>{value}</span>)}</div>
                </button>
              ))}
              {!official && <div className="model-browser-empty">{tr("正在读取官方目录…", "Loading the official catalog…")}</div>}
            </div>
            {selectedOfficial && selectedSource && (
              <aside className="model-detail-panel">
                <div className="model-detail-heading"><div><small>{selectedOfficial.family}</small><h3>{selectedOfficial.name}</h3></div><div className="model-identity-trailing"><ModelVendorMark name={selectedOfficial.name} architecture={selectedOfficial.architecture} size={30} /></div></div>
                <p>{tr(selectedOfficial.description_zh, selectedOfficial.description)}</p>
                <ConfigurationDetails status={officialConfiguration!} planned={officialCacheBytes > 0} tr={tr}
                  rows={variantMemoryPressureRows(plannedOfficialVariants[0], official?.system, officialCacheBytes,
                    (parameters?.routed_experts ?? 0) > 0 || selectedOfficial.capabilities.includes('MoE'))} />
                <dl className="model-metadata-grid">
                  <div><dt>{tr("架构", "Architecture")}</dt><dd>{selectedOfficial.architecture}</dd></div>
                  <div className="model-parameter-row"><dt>{tr("参数", "Parameters")}</dt><dd title={tr("主模型逻辑参数，含视觉模块；不含 MTP 辅助权重与量化元数据。", "Main-model logical parameters, including vision; excludes auxiliary MTP weights and quantization metadata.")}><ParameterValue parameters={parameters} label={selectedOfficial.parameter_label} tr={tr} /></dd></div>
                  <div><dt>{tr("激活参数", "Active parameters")}</dt><dd title={tr("主干每 token 的逻辑激活参数，不含词嵌入、输出词表、视觉编码器与 PLE 查表。", "Logical backbone parameters active per token; excludes vocabulary embeddings/output, vision encoder and PLE lookups.")}>{parameters?.active != null ? formatParameters(parameters.active) : selectedOfficial.active_parameter_label || "—"}</dd></div>
                  <div><dt>{tr("模态", "Modalities")}</dt><dd>{modalityLabels(selectedMetadata?.modalities?.length ? selectedMetadata.modalities : selectedOfficial.modalities, tr)}</dd></div>
                  <MtpSupport supported={mtpSupported} tr={tr} />
                  <div><dt>{tr("许可", "License")}</dt><dd>{selectedOfficial.license || tr("查看模型卡", "See model card")}</dd></div>
                  <div><dt>{tr("下载", "Downloads")}</dt><dd>{formatCount(selectedOfficial.downloads)}</dd></div>
                  <div><dt>{tr("收藏", "Likes")}</dt><dd>{formatCount(selectedOfficial.likes)}</dd></div>
                  <div><dt>{tr("发布时间", "Published")}</dt><dd>{formatDate(selectedOfficial.published_at || selectedOfficial.updated_at)}</dd></div>
                </dl>
                <KvCachePlanner key={`kv:${officialPlanKey}`} profile={selectedMetadata?.cache_profile} appliedContext={officialContext} onApply={(context) => applyPlan(officialPlanKey, context)} tr={tr} />
                <label className="model-source-picker"><span>{tr("下载来源", "Download source")}</span><select onChange={(event) => { const source = selectedOfficial.sources[Number(event.target.value)]; if (source) void chooseOfficialSource(source); }} value={String(Math.max(0, selectedOfficial.sources.findIndex((item) => item.provider === selectedSource.provider && item.repo_id === selectedSource.repo_id)))}>{selectedOfficial.sources.map((source, index) => <option disabled={!source.available} key={`${source.provider}:${source.repo_id}`} value={index}>{source.provider === "huggingface" ? "Hugging Face" : "ModelScope"}{source.available ? "" : ` · ${tr("离线", "unavailable")}`}</option>)}</select></label>
                <div className="repository-line"><button onClick={() => void openStudioExternal(selectedSource.url).catch((cause) => onError(cause instanceof Error ? cause.message : String(cause)))} type="button">{selectedSource.repo_id}</button><span>{officialVariants.length} {tr("个精度版本", "variants")}</span></div>
                <VariantList disabled={officialLoading || !selectedSource.available || !canDownload(selectedSource.provider) || downloading !== null} onDownload={(variant, origin) => void download({ provider: selectedSource.provider, repo_id: selectedSource.repo_id, revision: selectedSource.revision || selectedOfficial.revision }, variant, origin)} tr={tr} variants={plannedOfficialVariants} cacheBytes={officialCacheBytes} system={official?.system} moe={(parameters?.routed_experts ?? 0) > 0 || selectedOfficial.capabilities.includes('MoE')} />
                {sourceInfoMatches && <RepositoryFiles key={`${selectedSource.provider}:${selectedSource.repo_id}:${officialSourceInfo!.revision}`}
                  files={officialSourceInfo!.files} disabled={officialLoading || !canDownload(selectedSource.provider) || downloading !== null} tr={tr}
                  onDownload={(files, label, origin) => void download(officialSourceInfo!, {
                    id: label, label, format: 'unknown', files: files.map((file) => file.name),
                    byte_size: files.reduce((sum, file) => sum + file.byte_size, 0), configuration: { status: 'unknown', recommendation: 'unknown', reasons: [] },
                  }, origin)} />}
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
              {results.map((item) => <button className={communityModel?.repo_id === item.repo_id && communityModel.provider === item.provider ? "selected" : ""} key={`${item.provider}:${item.repo_id}`} onClick={() => void inspect(item)} type="button"><div><strong>{item.repo_id}</strong><small>{item.provider === "huggingface" ? "Hugging Face" : "ModelScope"} · {formatCount(item.downloads)} downloads</small></div><span className="model-identity-trailing">{formatBytes(item.total_bytes)}<ModelVendorMark name={item.repo_id} architecture={communityModel?.repo_id === item.repo_id && communityModel.provider === item.provider ? communityModel.architectures : undefined} /></span></button>)}
              {!results.length && <div className="model-browser-empty">{tr("搜索社区模型，或粘贴仓库链接直接打开。", "Search community models or paste a repository link to open it directly.")}</div>}
            </div>
            {communityModel && (
              <aside className="model-detail-panel">
                <div className="model-detail-heading"><div><small>{communityModel.author || communityModel.provider}</small><h3>{communityModel.repo_id.split("/").pop()}</h3></div><div className="model-identity-trailing">{communityModel.gated && <span className="gated-badge">{tr("需要授权", "Gated")}</span>}<ModelVendorMark name={communityModel.repo_id} architecture={communityModel.architectures} size={30} /></div></div>
                {communityModel.description && <p>{communityModel.description}</p>}
                <dl className="model-metadata-grid">
                  <div><dt>{tr("架构", "Architecture")}</dt><dd>{communityModel.architectures.join(", ") || tr("未声明", "Not declared")}</dd></div>
                  <div className="model-parameter-row"><dt>{tr("参数", "Parameters")}</dt><dd><ParameterValue parameters={communityModel.parameter_breakdown} label={communityModel.parameter_count ? formatParameters(communityModel.parameter_count) : null} tr={tr} /></dd></div>
                  <div><dt>{tr('激活参数', 'Active parameters')}</dt><dd>{communityModel.parameter_breakdown?.active != null ? formatParameters(communityModel.parameter_breakdown.active) : '—'}</dd></div>
                  <div><dt>{tr("模态", "Modalities")}</dt><dd>{modalityLabels(communityModel.modalities.length ? communityModel.modalities : ['text'], tr)}</dd></div>
                  <MtpSupport supported={communityModel.mtp_supported} tr={tr} />
                  <div><dt>{tr("许可", "License")}</dt><dd>{communityModel.license || tr("未声明", "Not declared")}</dd></div>
                  <div><dt>{tr("下载", "Downloads")}</dt><dd>{formatCount(communityModel.downloads)}</dd></div>
                  <div><dt>{tr("收藏", "Likes")}</dt><dd>{formatCount(communityModel.likes)}</dd></div>
                  <div><dt>{tr("发布时间", "Published")}</dt><dd>{formatDate(communityModel.published_at || communityModel.updated_at)}</dd></div>
                </dl>
                <KvCachePlanner key={`kv:${communityPlanKey}`} profile={communityModel.cache_profile} appliedContext={communityContext} onApply={(context) => applyPlan(communityPlanKey, context)} tr={tr} />
                {communityModel.source_url && <div className="repository-line"><button onClick={() => void openStudioExternal(communityModel.source_url!).catch((cause) => onError(cause instanceof Error ? cause.message : String(cause)))} type="button">{tr("打开模型卡", "Open model card")}</button><span>{formatCount(communityModel.downloads)} downloads · {formatCount(communityModel.likes)} likes · {communityModel.files.length} files</span></div>}
                <VariantList disabled={communityLoading || !canDownload(communityModel.provider) || downloading !== null} onDownload={(variant, origin) => void download({ provider: communityModel.provider, repo_id: communityModel.repo_id, revision: communityModel.revision }, variant, origin)} tr={tr} variants={withCacheMemory(communityModel.variants, communityCacheBytes)} cacheBytes={communityCacheBytes} system={official?.system} moe={(communityModel.parameter_breakdown?.routed_experts ?? 0) > 0} />
                <RepositoryFiles key={`${communityModel.provider}:${communityModel.repo_id}:${communityModel.revision}`} files={communityModel.files}
                  disabled={communityLoading || !canDownload(communityModel.provider) || downloading !== null} tr={tr}
                  onDownload={(files, label, origin) => void download(communityModel, {
                    id: label, label, format: 'unknown', files: files.map((file) => file.name),
                    byte_size: files.reduce((sum, file) => sum + file.byte_size, 0), configuration: { status: 'unknown', recommendation: 'unknown', reasons: [] },
                  }, origin)} />
              </aside>
            )}
          </div>
        </>
      ) : downloadQueue}
    </section>
  );
}
