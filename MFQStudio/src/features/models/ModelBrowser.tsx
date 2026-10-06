/** Browse model catalogs, compatibility details, and downloadable repository files. */
import { i18n } from '../../i18n';
import type { TFunction } from 'i18next';
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
  OfficialModelInfo,
  OfficialModelList,
  OfficialModelSource,
} from '../../shared/api/types';
import { jobsApi } from '../../shared/api/resources/jobs';
import { modelsApi } from '../../shared/api/resources/models';
import { openStudioExternal } from '../../shared/platform/studio';
import { BackendBadge } from './BackendBadge';
import { ModelVendorMark } from '../../app/ModelVendorMark';
import { RepositoryFiles } from './RepositoryFiles';

type Translate = TFunction;
export type ModelBrowserTab = 'official' | 'community' | 'downloads';
export type DownloadOrigin = { x: number; y: number };

interface ModelBrowserProps {
  jobKinds: JobKindResource[];
  onError(message: string): void;
  onJobCreated(job: JobResource, origin: DownloadOrigin): void;
  tab: ModelBrowserTab;
  onTabChange(tab: ModelBrowserTab): void;
  downloadQueue?: ReactNode;
  t: Translate;
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
  return new Intl.NumberFormat(i18n.resolvedLanguage, { notation: "compact" }).format(value);
}

function hardwareSummary(system: HubSystemProfile, t: Translate): string {
  const unified = system.memory_pools?.some((pool) => pool.kind === 'uma') || /^Apple M\d+(?: (?:Pro|Max|Ultra))?$/.test(system.cpu_name || '');
  const ram = system.physical_memory_bytes ? `${formatBytes(system.physical_memory_bytes).replace(/\.0 /, ' ')} ${unified ? 'URAM' : 'RAM'}` : null;
  const gpu = system.gpu_names?.join(' + ');
  if (system.backend === 'metal') {
    const cores = [system.cpu_cores && `${system.cpu_cores} CPU`, system.gpu_cores && `${system.gpu_cores} GPU`].filter(Boolean).join(' / ');
    return [system.cpu_name || gpu, cores, ram].filter(Boolean).join(' · ') || t('models:modelBrowser.hardwareDetailsUnavailable');
  }
  return [gpu, ram, system.cpu_name].filter(Boolean).join(' · ') || t('models:modelBrowser.hardwareDetailsUnavailable');
}

function MemoryBudget({ system, t }: { system: HubSystemProfile | undefined; t: Translate }) {
  const pools: HubMemoryPool[] = system?.memory_pools?.length ? system.memory_pools : system ? [{
    kind: /^Apple M\d+(?: (?:Pro|Max|Ultra))?$/.test(system.cpu_name || '') ? 'uma' : 'ram',
    capacity_bytes: system.physical_memory_bytes,
  }] : [];
  return <div className="detected-memory-pools">
    {pools.length ? pools.map((pool, index) => <div className="detected-memory-pool" key={`${pool.kind}:${index}`} title={pool.device || undefined}>
      <strong>{pool.capacity_bytes ? `${(pool.capacity_bytes / 2 ** 30).toFixed(1).replace(/\.0$/, '')} GiB` : '—'} {pool.kind === 'uma' ? 'URAM' : pool.kind.toUpperCase()}</strong>
      <small title={pool.bandwidth_bytes_per_second ? t('models:modelBrowser.specifiedBandwidthNotMeasuredThroughput') : undefined}>
        {pool.bandwidth_bytes_per_second ? `${new Intl.NumberFormat(i18n.resolvedLanguage, { maximumFractionDigits: 1 }).format(pool.bandwidth_bytes_per_second / 2 ** 30)} GiB/s` : t('models:modelBrowser.bandwidthUnavailable')}
      </small>
    </div>) : <strong>—</strong>}
  </div>;
}

function formatDate(value?: string | null): string {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleDateString(i18n.resolvedLanguage);
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

function configurationReasons(status: ModelConfigurationStatus, t: Translate): string[] {
  return status.reasons.map((reason) => {
    if (reason === "Configuration requirements could not be determined.") return t('models:modelBrowser.configurationRequirementsCouldNotBeDetermined');
    if (reason === "Estimated memory requirement exceeds the detected runtime budget.") return t('models:modelBrowser.estimatedMemoryRequirementExceedsTheDetectedRuntimeBudget');
    if (reason === "Estimated memory requirement exceeds the detected runtime budget, but at least 70% is available.") return t('models:modelBrowser.estimatedMemoryRequirementExceedsTheDetectedRuntimeBudgetButAtLeast70');
    if (reason === "The detected runtime budget is below 70% of the estimated memory requirement.") return t('models:modelBrowser.theDetectedRuntimeBudgetIsBelow70OfTheEstimatedMemoryRequirement');
    if (reason === "Fits within the detected runtime memory budget.") return t('models:modelBrowser.fitsWithinTheDetectedRuntimeMemoryBudget');
    if (reason === "Close other memory-heavy applications before loading.") return t('models:modelBrowser.closeOtherMemoryHeavyApplicationsBeforeLoading');
    if (reason === "All published precision tiers fit within the detected runtime memory budget.") return t('models:modelBrowser.allPublishedPrecisionTiersFitWithinTheDetectedRuntimeMemoryBudget');
    if (reason === "More than half of the published precision tiers fit within the detected runtime memory budget.") return t('models:modelBrowser.moreThanHalfOfThePublishedPrecisionTiersFitWithinTheDetected');
    if (reason === "At most half of the published precision tiers fit within the detected runtime memory budget.") return t('models:modelBrowser.atMostHalfOfThePublishedPrecisionTiersFitWithinTheDetected');
    if (reason === "The detected runtime memory budget covers at least 70% of the smallest published precision tier.") return t('models:modelBrowser.theDetectedRuntimeMemoryBudgetCoversAtLeast70OfTheSmallest');
    if (reason === "The detected runtime memory budget is below 70% of the smallest published precision tier.") return t('models:modelBrowser.theDetectedRuntimeMemoryBudgetIsBelow70OfTheSmallestPublished');
    if (reason === "All published precision tiers fit fully within the detected runtime memory budget.") return t('models:modelBrowser.allPublishedPrecisionTiersFitFullyWithinTheDetectedRuntimeMemoryBudget');
    if (reason === "More than half of the published precision tiers fit fully within the detected runtime memory budget.") return t('models:modelBrowser.moreThanHalfOfThePublishedPrecisionTiersFitFullyWithinThe');
    if (reason === "At most half of the published precision tiers fit fully within the detected runtime memory budget.") return t('models:modelBrowser.atMostHalfOfThePublishedPrecisionTiersFitFullyWithinThe');
    if (reason === "The detected runtime memory budget covers at least 70% of the full-residency requirement for the smallest published precision tier.") return t('models:modelBrowser.theDetectedRuntimeMemoryBudgetCoversAtLeast70OfTheFull');
    if (reason === "The detected runtime memory budget is below 70% of the full-residency requirement for the smallest published precision tier.") return t('models:modelBrowser.theDetectedRuntimeMemoryBudgetIsBelow70OfTheFullResidency');
    if (reason === "The catalog is available offline; repository metadata could not be refreshed.") return t('models:modelBrowser.theCatalogIsAvailableOfflineRepositoryMetadataCouldNotBeRefreshed');
    if (reason === "GGUF must be converted before the MFQ runtime can load it.") return t('models:modelBrowser.ggufMustBeConvertedBeforeTheMfqRuntimeCanLoadIt');
    if (reason === "The repository can be downloaded, but direct runtime compatibility has not been verified.") return t('models:modelBrowser.theRepositoryCanBeDownloadedButDirectRuntimeCompatibilityHasNotBeen');
    if (reason === "This format is not directly loadable by MFQ.") return t('models:modelBrowser.thisFormatIsNotDirectlyLoadableByMfq');
    if (reason === "This model architecture is not registered in MFQ.") return t('models:modelBrowser.thisModelArchitectureIsNotRegisteredInMfq');
    if (reason === "Runtime compatibility could not be verified from repository metadata.") return t('models:modelBrowser.runtimeCompatibilityCouldNotBeVerifiedFromRepositoryMetadata');
    if (reason === "Tensor payload baseline; excludes streamed PLE, KV cache and runtime repacking.") return t('models:modelBrowser.tensorPayloadBaselineExcludesStreamedPleKvCacheAndRuntimeRepacking');
    if (reason === "File-size estimate; tensor metadata is unavailable.") return t('models:modelBrowser.fileSizeEstimateTensorMetadataIsUnavailable');
    if (reason === "All published precision tiers' weight baselines fit within the detected runtime memory budget.") return t('models:modelBrowser.allPublishedPrecisionTiersWeightBaselinesFitWithinTheDetectedRuntimeMemory');
    if (reason === "More than half of the published precision tiers' weight baselines fit within the detected runtime memory budget.") return t('models:modelBrowser.moreThanHalfOfThePublishedPrecisionTiersWeightBaselinesFitWithin');
    if (reason === "At most half of the published precision tiers' weight baselines fit within the detected runtime memory budget.") return t('models:modelBrowser.atMostHalfOfThePublishedPrecisionTiersWeightBaselinesFitWithin');
    if (reason === "The detected runtime memory budget covers at least 70% of the weight baseline requirement for the smallest published precision tier.") return t('models:modelBrowser.theDetectedRuntimeMemoryBudgetCoversAtLeast70OfTheWeight');
    if (reason === "The detected runtime memory budget is below 70% of the weight baseline requirement for the smallest published precision tier.") return t('models:modelBrowser.theDetectedRuntimeMemoryBudgetIsBelow70OfTheWeightBaseline');
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

function ConfigurationBadge({ status, t }: { status: ModelConfigurationStatus; t: Translate }) {
  const reason = configurationReasons(status, t).join(" ");
  const presentation = {
    three_stars: ["★★★", t('models:modelBrowser.lowWeightPressureAllTierBaselinesFitTheBudget')],
    two_stars: ["★★", t('models:modelBrowser.moderateWeightPressureMostTierBaselinesFitTheBudget')],
    one_star: ["★", t('models:modelBrowser.highWeightPressureOnlySomeTierBaselinesFitTheBudget')],
    caution: ["▲", t('models:modelBrowser.weightBudgetNearLimitTheSmallestTierBaselineNearlyFits')],
    not_recommended: ["✕", t('models:modelBrowser.insufficientWeightBudgetTheSmallestTierBaselineExceedsIt')],
    unknown: ["?", t('models:modelBrowser.memoryPressureUnknown')],
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

function ConfigurationDetails({ status, t }: { status: ModelConfigurationStatus; t: Translate }) {
  return (
    <div className={`configuration-details ${status.status} ${status.recommendation.replaceAll("_", "-")}`}>
      <div>
        <span>{t('models:modelBrowser.smallestTierWeightBaseline')}</span>
        <strong>{formatBytes(status.required_memory_bytes)}</strong>
      </div>
      <div>
        <span>{t('models:modelBrowser.detectedRuntimeBudget')}</span>
        <strong>{formatBytes(status.available_memory_bytes)}</strong>
      </div>
      <p>{configurationReasons(status, t).join(" ")}</p>
    </div>
  );
}

function VariantList({
  disabled,
  onDownload,
  t,
  variants,
}: {
  disabled: boolean;
  onDownload(variant: HubModelVariant, origin: DownloadOrigin): void;
  t: Translate;
  variants: HubModelVariant[];
}) {
  if (!variants.length) {
    return <div className="model-browser-empty compact">{t('models:modelBrowser.noDownloadableWeightsWereReturnedByThisRepository')}</div>;
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
              <small>{variant.precision || variant.format.toUpperCase()} · {t('models:modelBrowser.file')} {formatBytes(variant.byte_size)} · {variant.resident_weight_bytes != null ? t('models:modelBrowser.residentWeightBaseline') : t('models:modelBrowser.estWeightMemory')} {formatBytes(variant.configuration.required_memory_bytes)}{(variant.ssd_ple_bytes ?? 0) > 0 && ` · SSD PLE ${formatBytes(variant.ssd_ple_bytes)}`}</small>
            </div>
            <div className="variant-memory-pressure">
              <div
                aria-label={`${t('models:modelBrowser.estimatedShareOfRuntimeBudget')}: ${percentageLabel}`}
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
              {t('models:modelBrowser.download')}
            </button>
          </div>
        );
      })}
    </div>
  );
}

/** Browse model catalogs, compatibility details, and downloadable repository files. */
export function ModelBrowser({ jobKinds, onError, onJobCreated, tab, onTabChange, downloadQueue, t }: ModelBrowserProps) {
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
          <h2>{t('models:modelBrowser.modelBrowser')}</h2>
          <p>{t('models:modelBrowser.findOptimizedOfficialModelsOrBrowseCommunityRepositories')}</p>
        </div>
        <div className="model-browser-actions">
          <label className="download-concurrency">{t('models:modelBrowser.maximumConcurrentDownloads')}
            <select aria-label={t('models:modelBrowser.maximumConcurrentDownloads')} value={maxWorkers} onChange={(event) => setMaxWorkers(Number(event.target.value))}>
              {[1, 2, 4, 8, 16].map((count) => <option key={count} value={count}>{count}</option>)}
            </select>
          </label>
        <div className="model-browser-tabs" role="tablist">
          <button aria-selected={tab === "official"} onClick={() => onTabChange("official")} role="tab" type="button">{t('models:modelBrowser.official')}</button>
          <button aria-selected={tab === "community"} onClick={() => onTabChange("community")} role="tab" type="button">{t('models:modelBrowser.community')}</button>
          <button aria-selected={tab === "downloads"} onClick={() => onTabChange("downloads")} role="tab" type="button">{t('models:modelBrowser.downloadQueue')}</button>
        </div>
        </div>
      </header>

      {tab === "official" ? (
        <>
          <div className="detected-configuration">
            <div><span>{t('models:modelBrowser.detectedConfiguration')}</span><div className="detected-hardware-summary"><strong>{system ? hardwareSummary(system, t) : t('models:modelBrowser.detecting')}</strong>{system && <BackendBadge backend={system.backend} />}</div></div>
            <div><span>{t('models:modelBrowser.runtimeBudget')}</span><MemoryBudget system={system || undefined} t={t} /></div>
            <button disabled={catalogLoading || official?.refreshing} onClick={() => void loadOfficial(true)} type="button">{catalogLoading || official?.refreshing ? t('models:modelBrowser.refreshing') : t('models:modelBrowser.refresh')}</button>
          </div>
          <div className="memory-pressure-guide">
            <strong>{t('models:modelBrowser.memoryPressure')}</strong>
            <span><b>★★★</b>{t('models:modelBrowser.low')}</span>
            <span><b>★★</b>{t('models:modelBrowser.moderate')}</span>
            <span><b>★</b>{t('models:modelBrowser.high')}</span>
            <span><b>▲</b>{t('models:modelBrowser.nearLimit')}</span>
            <span><b>✕</b>{t('models:modelBrowser.insufficient')}</span>
            <small>{t('models:modelBrowser.basedOnResidentWeightBaselinesKvCacheAndRuntimeOverheadNeedAdditional')}</small>
          </div>
          <div className="model-browser-layout">
            <div className="official-model-grid">
              {official?.data.map((item) => (
                <button className={selectedOfficial?.id === item.id ? "official-model-card selected" : "official-model-card"} key={item.id} onClick={() => chooseOfficial(item)} type="button">
                  <div className="official-model-card-title"><span>{item.name.slice(0, 1).toUpperCase()}</span><div><strong>{item.name}</strong><small>{item.family}</small></div><ModelVendorMark name={item.name} architecture={item.architecture} size={28} /></div>
                  <p>{(i18n.resolvedLanguage === 'zh-CN' ? item.description_zh || item.description : item.description || item.description_zh)}</p>
                  <div className="model-chip-row">{item.precision_options.map((value) => <span key={value}>{value}</span>)}</div>
                  <ConfigurationBadge status={item.configuration} t={t} />
                </button>
              ))}
              {!official && <div className="model-browser-empty">{t('models:modelBrowser.loadingTheOfficialCatalog')}</div>}
            </div>
            {selectedOfficial && selectedSource && (
              <aside className="model-detail-panel">
                <div className="model-detail-heading"><div><small>{selectedOfficial.family}</small><h3>{selectedOfficial.name}</h3></div><div className="model-identity-trailing"><ConfigurationBadge status={selectedOfficial.configuration} t={t} /><ModelVendorMark name={selectedOfficial.name} architecture={selectedOfficial.architecture} size={30} /></div></div>
                <p>{(i18n.resolvedLanguage === 'zh-CN' ? selectedOfficial.description_zh || selectedOfficial.description : selectedOfficial.description || selectedOfficial.description_zh)}</p>
                <ConfigurationDetails status={selectedOfficial.configuration} t={t} />
                <dl className="model-metadata-grid">
                  <div><dt>{t('models:modelBrowser.architecture')}</dt><dd>{selectedOfficial.architecture}</dd></div>
                  <div><dt>{t('models:modelBrowser.parameters')}</dt><dd>{selectedOfficial.parameter_label || "—"}</dd></div>
                  <div><dt>{t('models:modelBrowser.activeParameters')}</dt><dd>{selectedOfficial.active_parameter_label || "—"}</dd></div>
                  <div><dt>{t('models:modelBrowser.modalities')}</dt><dd>{selectedOfficial.modalities.join(" · ")}</dd></div>
                  <div><dt>{t('models:modelBrowser.capabilities')}</dt><dd>{selectedOfficial.capabilities.join(" · ")}</dd></div>
                  <div><dt>{t('models:modelBrowser.license')}</dt><dd>{selectedOfficial.license || t('models:modelBrowser.seeModelCard')}</dd></div>
                  <div><dt>{t('models:modelBrowser.downloads')}</dt><dd>{formatCount(selectedOfficial.downloads)}</dd></div>
                  <div><dt>{t('models:modelBrowser.likes')}</dt><dd>{formatCount(selectedOfficial.likes)}</dd></div>
                  <div><dt>{t('models:modelBrowser.published')}</dt><dd>{formatDate(selectedOfficial.published_at || selectedOfficial.updated_at)}</dd></div>
                </dl>
                <label className="model-source-picker"><span>{t('models:modelBrowser.downloadSource')}</span><select onChange={(event) => { const source = selectedOfficial.sources[Number(event.target.value)]; if (source) void chooseOfficialSource(source); }} value={String(Math.max(0, selectedOfficial.sources.findIndex((item) => item.provider === selectedSource.provider && item.repo_id === selectedSource.repo_id)))}>{selectedOfficial.sources.map((source, index) => <option disabled={!source.available} key={`${source.provider}:${source.repo_id}`} value={index}>{source.provider === "huggingface" ? "Hugging Face" : "ModelScope"}{source.available ? "" : ` · ${t('models:modelBrowser.unavailable')}`}</option>)}</select></label>
                <div className="repository-line"><button onClick={() => void openStudioExternal(selectedSource.url).catch((cause) => onError(cause instanceof Error ? cause.message : String(cause)))} type="button">{selectedSource.repo_id}</button><span>{officialVariants.length} {t('models:modelBrowser.variants')}</span></div>
                {selectedOfficial.supports_ssd_streaming && <div className="streaming-note">{t('models:modelBrowser.ssdExpertStreamingRemainsAvailableWhenTheModelCannotFitFullyIn')}</div>}
                <VariantList disabled={officialLoading || !selectedSource.available || !canDownload(selectedSource.provider) || downloading !== null} onDownload={(variant, origin) => void download({ provider: selectedSource.provider, repo_id: selectedSource.repo_id, revision: selectedSource.revision || selectedOfficial.revision }, variant, origin)} t={t} variants={officialVariants} />
                {sourceInfoMatches && <RepositoryFiles key={`${selectedSource.provider}:${selectedSource.repo_id}:${officialSourceInfo!.revision}`}
                  files={officialSourceInfo!.files} disabled={officialLoading || !canDownload(selectedSource.provider) || downloading !== null} t={t}
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
            <input onChange={(event) => setQuery(event.target.value)} placeholder={t('models:modelBrowser.modelNameOwnerRepoOrRepositoryUrl')} value={query} />
            <button disabled={communityLoading || !query.trim()} type="submit">{communityLoading ? t('models:modelBrowser.searching') : t('models:modelBrowser.search')}</button>
          </form>
          <p className="community-search-hint">{t('models:modelBrowser.pasteAHuggingFaceOrModelscopeLinkDirectlyDownloadsRemainResumableBackground')}</p>
          <div className="model-browser-layout community-layout">
            <div className="community-results">
              {results.map((item) => <button className={communityModel?.repo_id === item.repo_id && communityModel.provider === item.provider ? "selected" : ""} key={`${item.provider}:${item.repo_id}`} onClick={() => void inspect(item)} type="button"><div><strong>{item.repo_id}</strong><small>{item.provider === "huggingface" ? "Hugging Face" : "ModelScope"} · {formatCount(item.downloads)} downloads</small></div><span className="model-identity-trailing">{formatBytes(item.total_bytes)}<ModelVendorMark name={item.repo_id} architecture={communityModel?.repo_id === item.repo_id && communityModel.provider === item.provider ? communityModel.architectures : undefined} /></span></button>)}
              {!results.length && <div className="model-browser-empty">{t('models:modelBrowser.searchCommunityModelsOrPasteARepositoryLinkToOpenItDirectly')}</div>}
            </div>
            {communityModel && (
              <aside className="model-detail-panel">
                <div className="model-detail-heading"><div><small>{communityModel.author || communityModel.provider}</small><h3>{communityModel.repo_id.split("/").pop()}</h3></div><div className="model-identity-trailing">{communityModel.gated && <span className="gated-badge">{t('models:modelBrowser.gated')}</span>}<ModelVendorMark name={communityModel.repo_id} architecture={communityModel.architectures} size={30} /></div></div>
                {communityModel.description && <p>{communityModel.description}</p>}
                <dl className="model-metadata-grid">
                  <div><dt>{t('models:modelBrowser.architecture')}</dt><dd>{communityModel.architectures.join(", ") || t('models:modelBrowser.notDeclared')}</dd></div>
                  <div><dt>{t('models:modelBrowser.parameters')}</dt><dd>{communityModel.parameter_count ? formatCount(communityModel.parameter_count) : "—"}</dd></div>
                  <div><dt>{t('models:modelBrowser.modalities')}</dt><dd>{communityModel.modalities.join(" · ") || "text"}</dd></div>
                  <div><dt>{t('models:modelBrowser.license')}</dt><dd>{communityModel.license || t('models:modelBrowser.notDeclared')}</dd></div>
                  <div><dt>{t('models:modelBrowser.repositorySize')}</dt><dd>{formatBytes(communityModel.total_bytes)}</dd></div>
                  <div><dt>{t('models:modelBrowser.revision')}</dt><dd title={communityModel.revision}>{communityModel.revision.slice(0, 12)}</dd></div>
                  <div><dt>{t('models:modelBrowser.library')}</dt><dd>{communityModel.library || "—"}</dd></div>
                  <div><dt>{t('models:modelBrowser.task')}</dt><dd>{communityModel.pipeline_tag || "—"}</dd></div>
                  <div><dt>{t('models:modelBrowser.updated')}</dt><dd>{formatDate(communityModel.updated_at)}</dd></div>
                  <div><dt>{t('models:modelBrowser.mfqCompatibility')}</dt><dd>{communityModel.runtime_compatible === true ? t('models:modelBrowser.verified') : communityModel.runtime_compatible === false ? t('models:modelBrowser.unsupported') : t('models:modelBrowser.unknown')}</dd></div>
                </dl>
                {communityModel.source_url && <div className="repository-line"><button onClick={() => void openStudioExternal(communityModel.source_url!).catch((cause) => onError(cause instanceof Error ? cause.message : String(cause)))} type="button">{t('models:modelBrowser.openModelCard')}</button><span>{formatCount(communityModel.downloads)} downloads · {formatCount(communityModel.likes)} likes · {communityModel.files.length} files</span></div>}
                <VariantList disabled={communityLoading || !canDownload(communityModel.provider) || downloading !== null} onDownload={(variant, origin) => void download({ provider: communityModel.provider, repo_id: communityModel.repo_id, revision: communityModel.revision }, variant, origin)} t={t} variants={communityModel.variants} />
                <RepositoryFiles key={`${communityModel.provider}:${communityModel.repo_id}:${communityModel.revision}`} files={communityModel.files}
                  disabled={communityLoading || !canDownload(communityModel.provider) || downloading !== null} t={t}
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
