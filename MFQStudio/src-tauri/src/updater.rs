use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
#[cfg(target_os = "macos")]
use std::fs::OpenOptions;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
#[cfg(any(target_os = "macos", target_os = "windows"))]
use std::process::Command;
#[cfg(target_os = "macos")]
use std::process::Stdio;
use std::sync::{Arc, Mutex as StdMutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tauri::{AppHandle, Manager, State};
use tokio::sync::Mutex;
use url::Url;

const RELEASES_URL: &str = "https://api.github.com/repos/Tylogi/TyloQuant/releases?per_page=100";
const RELEASES_PAGE: &str = "https://github.com/Tylogi/TyloQuant/releases";
const CACHE_SECONDS: u64 = 6 * 60 * 60;
const PREFERENCES_FILE: &str = "update-preferences.json";
const RELEASES_FILE: &str = "releases.json";
const INSTALL_METADATA_FILE: &str = "release.json";
const STUDIO_CONFIG_FILE: &str = "studio.json";
const SERVER_PID_FILE: &str = "mfq-server.pid";
const HELPER_ARGUMENT: &str = "--mfq-update-helper";
#[cfg(target_os = "macos")]
const BUNDLE_IDENTIFIER: &str = "com.tylogi.mfq-studio";

pub struct UpdateState {
    lock: Mutex<()>,
    progress: Arc<StdMutex<Option<UpdateProgress>>>,
}

impl Default for UpdateState {
    fn default() -> Self {
        Self {
            lock: Mutex::new(()),
            progress: Arc::new(StdMutex::new(None)),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct UpdatePreferences {
    automatic_check: bool,
    #[serde(default)]
    automatic_download: bool,
}

impl Default for UpdatePreferences {
    fn default() -> Self {
        Self {
            automatic_check: true,
            automatic_download: false,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct GitHubAsset {
    name: String,
    browser_download_url: String,
    size: u64,
    content_type: String,
    digest: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct GitHubRelease {
    tag_name: String,
    name: Option<String>,
    body: Option<String>,
    html_url: String,
    published_at: Option<String>,
    draft: bool,
    prerelease: bool,
    assets: Vec<GitHubAsset>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StudioReleaseAsset {
    name: String,
    byte_size: u64,
    sha256: Option<String>,
    download_url: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StudioRelease {
    version: String,
    tag: String,
    name: String,
    notes: String,
    published_at: Option<String>,
    page_url: String,
    prerelease: bool,
    asset: Option<StudioReleaseAsset>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct InstalledVersion {
    version: String,
    tag: String,
    current: bool,
    ready: bool,
    byte_size: u64,
    installed_at_epoch_seconds: u64,
    notes: String,
    prerelease: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct UpdateStatus {
    current_version: String,
    current_release: bool,
    automatic_check: bool,
    automatic_download: bool,
    checked_at_epoch_seconds: Option<u64>,
    update_available: bool,
    platform_supported: bool,
    releases_page: &'static str,
    latest: Option<StudioRelease>,
    releases: Vec<StudioRelease>,
    installed_versions: Vec<InstalledVersion>,
    error: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct UpdateProgress {
    tag: String,
    stage: String,
    received_bytes: u64,
    total_bytes: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ReleasesCache {
    checked_at_epoch_seconds: u64,
    releases: Vec<StudioRelease>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct InstalledMetadata {
    release: StudioRelease,
    installed_at_epoch_seconds: u64,
}

#[derive(Debug, Deserialize)]
struct StudioConnectionConfig {
    local_service_port: u16,
}

fn now_epoch_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn update_root(app: &AppHandle) -> Result<PathBuf, String> {
    let root = app
        .path()
        .app_local_data_dir()
        .map_err(|error| error.to_string())?
        .join("updates");
    fs::create_dir_all(&root).map_err(|error| error.to_string())?;
    Ok(root)
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Option<T> {
    serde_json::from_slice(&fs::read(path).ok()?).ok()
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or_else(|| "update metadata has no parent directory".to_string())?;
    fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    let temporary = path.with_extension("json.partial");
    let bytes = serde_json::to_vec_pretty(value).map_err(|error| error.to_string())?;
    fs::write(&temporary, bytes).map_err(|error| error.to_string())?;
    if path.exists() {
        fs::remove_file(path).map_err(|error| error.to_string())?;
    }
    fs::rename(temporary, path).map_err(|error| error.to_string())
}

fn preferences(root: &Path) -> UpdatePreferences {
    read_json(&root.join(PREFERENCES_FILE)).unwrap_or_default()
}

fn extract_version(value: &str) -> Option<String> {
    let start = value.find(|character: char| character.is_ascii_digit())?;
    let candidate: String = value[start..]
        .chars()
        .take_while(|character| character.is_ascii_digit() || *character == '.')
        .collect();
    let components: Vec<_> = candidate.split('.').collect();
    if components.len() < 3 || components.iter().any(|item| item.parse::<u64>().is_err()) {
        return None;
    }
    Some(components[..3].join("."))
}

fn is_release_tag(tag: &str) -> bool {
    let Some(version) = extract_version(tag) else { return false; };
    let normalized = tag.to_ascii_lowercase();
    let numeric = normalized.strip_prefix('v').unwrap_or(&normalized);
    if numeric == version { return true; }
    let prefix = format!("studio-v{version}-multimodal-");
    normalized.strip_prefix(&prefix).is_some_and(|date|
        date.len() == 8 && date.chars().all(|item| item.is_ascii_digit()))
}

fn version_components(value: &str) -> Option<[u64; 3]> {
    let version = extract_version(value)?;
    let values: Vec<u64> = version
        .split('.')
        .map(str::parse)
        .collect::<Result<_, _>>()
        .ok()?;
    Some([values[0], values[1], values[2]])
}

fn compare_versions(left: &str, right: &str) -> Ordering {
    match (version_components(left), version_components(right)) {
        (Some(left), Some(right)) => left.cmp(&right),
        _ => left.cmp(right),
    }
}

fn asset_for_platform(release: &GitHubRelease) -> Option<&GitHubAsset> {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    return release.assets.iter().find(|asset| {
        let name = asset.name.to_ascii_lowercase();
        name.ends_with(".dmg") && (name.contains("aarch64") || name.contains("arm64"))
    });
    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
    return release.assets.iter().find(|asset| {
        let name = asset.name.to_ascii_lowercase();
        name.ends_with(".exe") && name.contains("x64")
    });
    #[allow(unreachable_code)]
    None
}

fn normalized_release(release: GitHubRelease) -> Option<StudioRelease> {
    let normalized_tag = release.tag_name.to_ascii_lowercase();
    if release.draft
        || release.prerelease
        || ["alpha", "beta", "dev", "rc", "nightly"]
            .iter()
            .any(|value| normalized_tag.contains(value))
    {
        return None;
    }
    let version = extract_version(&release.tag_name)?;
    if !is_release_tag(&release.tag_name) {
        return None;
    }
    let asset = asset_for_platform(&release).and_then(|asset| {
        let sha256 = asset.digest.as_deref()
            .and_then(|value| value.strip_prefix("sha256:"))
            .filter(|value| value.len() == 64 && value.chars().all(|item| item.is_ascii_hexdigit()))
            .map(str::to_ascii_lowercase)?;
        validate_download_url(&asset.browser_download_url).ok()?;
        Some(StudioReleaseAsset {
            name: asset.name.clone(), byte_size: asset.size,
            sha256: Some(sha256), download_url: asset.browser_download_url.clone(),
        })
    });
    Some(StudioRelease {
        version,
        tag: release.tag_name.clone(),
        name: release.name.unwrap_or_else(|| release.tag_name.clone()),
        notes: release.body.unwrap_or_default(),
        published_at: release.published_at,
        page_url: release.html_url,
        prerelease: release.prerelease,
        asset,
    })
}

fn fetch_releases() -> Result<Vec<StudioRelease>, String> {
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(20))
        .user_agent(format!("MFQ-Studio/{}", env!("CARGO_PKG_VERSION")))
        .build()
        .map_err(|error| error.to_string())?;
    let response = client
        .get(RELEASES_URL)
        .send()
        .and_then(reqwest::blocking::Response::error_for_status)
        .map_err(|error| format!("GitHub release check failed: {error}"))?;
    let releases: Vec<GitHubRelease> = response
        .json()
        .map_err(|error| format!("GitHub release response was invalid: {error}"))?;
    let mut releases: Vec<_> = releases
        .into_iter()
        .filter_map(normalized_release)
        .collect();
    releases.sort_by(|left, right| compare_versions(&right.version, &left.version));
    releases.dedup_by(|left, right| left.version == right.version);
    Ok(releases)
}

fn releases(root: &Path, force: bool, automatic: bool) -> (ReleasesCache, Option<String>) {
    let path = root.join(RELEASES_FILE);
    let cached: Option<ReleasesCache> = read_json(&path).map(|mut cache: ReleasesCache| {
        cache.releases.retain(|release| !release.prerelease && is_release_tag(&release.tag));
        cache
    });
    let fresh = cached.as_ref().is_some_and(|value| {
        now_epoch_seconds().saturating_sub(value.checked_at_epoch_seconds) < CACHE_SECONDS
    });
    if !force && (!automatic || fresh) {
        return (
            cached.unwrap_or(ReleasesCache {
                checked_at_epoch_seconds: 0,
                releases: Vec::new(),
            }),
            None,
        );
    }
    match fetch_releases() {
        Ok(values) => {
            let result = ReleasesCache {
                checked_at_epoch_seconds: now_epoch_seconds(),
                releases: values,
            };
            if let Err(error) = write_json(&path, &result) {
                return (
                    result,
                    Some(format!("Could not cache release metadata: {error}")),
                );
            }
            (result, None)
        }
        Err(error) => (
            cached.unwrap_or(ReleasesCache {
                checked_at_epoch_seconds: 0,
                releases: Vec::new(),
            }),
            Some(error),
        ),
    }
}

fn safe_version_directory(version: &str) -> Result<String, String> {
    let normalized =
        extract_version(version).ok_or_else(|| "invalid release version".to_string())?;
    let development_suffix = version.strip_prefix(&format!("{normalized}+dev."));
    if normalized != version && !development_suffix.is_some_and(|suffix|
        !suffix.is_empty() && suffix.chars().all(|item| item.is_ascii_alphanumeric() || item == '.')) {
        return Err("release version is not normalized".into());
    }
    Ok(version.to_string())
}

fn directory_size(path: &Path) -> u64 {
    let Ok(entries) = fs::read_dir(path) else {
        return 0;
    };
    entries
        .flatten()
        .map(|entry| {
            let path = entry.path();
            let Ok(file_type) = entry.file_type() else {
                return 0;
            };
            if file_type.is_dir() {
                directory_size(&path)
            } else if file_type.is_file() {
                entry.metadata().map(|value| value.len()).unwrap_or(0)
            } else {
                0
            }
        })
        .sum()
}

fn installed_versions(root: &Path) -> Vec<InstalledVersion> {
    let current_version = env!("MFQ_STUDIO_VERSION").to_string();
    let mut values = vec![InstalledVersion {
        version: current_version.clone(),
        tag: format!("v{current_version}"),
        current: true,
        ready: true,
        byte_size: 0,
        installed_at_epoch_seconds: 0,
        notes: String::new(),
        prerelease: env!("MFQ_STUDIO_RELEASE") != "true",
    }];
    let versions_root = root.join("versions");
    if let Ok(entries) = fs::read_dir(&versions_root) {
        for entry in entries.flatten() {
            let directory = entry.path();
            let Some(metadata) =
                read_json::<InstalledMetadata>(&directory.join(INSTALL_METADATA_FILE))
            else {
                continue;
            };
            let ready = cached_installable(&directory).is_some();
            if metadata.release.version == current_version {
                if let Some(current) = values.first_mut() {
                    current.byte_size = directory_size(&directory);
                    current.installed_at_epoch_seconds = metadata.installed_at_epoch_seconds;
                    current.notes = metadata.release.notes.clone();
                }
                continue;
            }
            values.push(InstalledVersion {
                version: metadata.release.version,
                tag: metadata.release.tag,
                current: false,
                ready,
                byte_size: directory_size(&directory),
                installed_at_epoch_seconds: metadata.installed_at_epoch_seconds,
                notes: metadata.release.notes,
                prerelease: metadata.release.prerelease,
            });
        }
    }
    values.sort_by(|left, right| {
        right
            .current
            .cmp(&left.current)
            .then_with(|| compare_versions(&right.version, &left.version))
    });
    values
}

fn cached_installable(directory: &Path) -> Option<PathBuf> {
    #[cfg(target_os = "macos")]
    {
        let path = directory.join("MFQ Studio.app");
        return path.is_dir().then_some(path);
    }
    #[cfg(target_os = "windows")]
    {
        return fs::read_dir(directory)
            .ok()?
            .flatten()
            .map(|item| item.path())
            .find(|path| {
                path.extension()
                    .and_then(|value| value.to_str())
                    .is_some_and(|value| value.eq_ignore_ascii_case("exe"))
            });
    }
    #[allow(unreachable_code)]
    None
}

fn update_status(root: &Path, force: bool) -> UpdateStatus {
    let preferences = preferences(root);
    let (cache, error) = releases(root, force, preferences.automatic_check);
    let current_version = env!("MFQ_STUDIO_VERSION").to_string();
    let latest = cache.releases.first().cloned();
    let update_available = latest
        .as_ref()
        .is_some_and(|value| compare_versions(&value.version, &current_version).is_gt());
    UpdateStatus {
        current_version,
        current_release: env!("MFQ_STUDIO_RELEASE") == "true",
        automatic_check: preferences.automatic_check,
        automatic_download: preferences.automatic_download,
        checked_at_epoch_seconds: (cache.checked_at_epoch_seconds > 0)
            .then_some(cache.checked_at_epoch_seconds),
        update_available,
        platform_supported: cfg!(all(target_os = "macos", target_arch = "aarch64"))
            || cfg!(all(target_os = "windows", target_arch = "x86_64")),
        releases_page: RELEASES_PAGE,
        latest,
        releases: cache.releases,
        installed_versions: installed_versions(root),
        error,
    }
}

fn validate_download_url(value: &str) -> Result<(), String> {
    let url = Url::parse(value).map_err(|error| format!("invalid release asset URL: {error}"))?;
    if url.scheme() != "https"
        || url.host_str() != Some("github.com")
        || !url.username().is_empty()
        || url.password().is_some()
        || url.port_or_known_default() != Some(443)
    {
        return Err("release assets must originate from github.com over HTTPS".into());
    }
    let trusted_prefixes = [
        "/Tylogi/TyloQuant/releases/download/",
        "/Tylogi/MFQ/releases/download/",
    ];
    if !trusted_prefixes
        .iter()
        .any(|prefix| url.path().starts_with(prefix))
    {
        return Err("release asset does not belong to Tylogi/TyloQuant".into());
    }
    Ok(())
}

fn safe_asset_name(value: &str) -> Result<&str, String> {
    let path = Path::new(value);
    if value.is_empty()
        || value.contains(['/', '\\', ':'])
        || path.components().count() != 1
        || path.file_name().and_then(|item| item.to_str()) != Some(value)
    {
        return Err("release asset name is unsafe".into());
    }
    Ok(value)
}

fn download_asset<F>(
    root: &Path,
    release: &StudioRelease,
    mut report_progress: F,
) -> Result<PathBuf, String>
where
    F: FnMut(u64, u64),
{
    let asset = release.asset.as_ref().ok_or_else(|| "this release has no verified installer for this platform".to_string())?;
    validate_download_url(&asset.download_url)?;
    let version = safe_version_directory(&release.version)?;
    if version != extract_version(&version).unwrap_or_default() {
        return Err("experimental versions cannot use Release downloads".into());
    }
    let name = safe_asset_name(&asset.name)?;
    let expected_sha256 = asset
        .sha256
        .as_deref()
        .ok_or_else(|| "release asset does not publish a SHA-256 digest".to_string())?;
    if expected_sha256.len() != 64 || !expected_sha256.bytes().all(|value| value.is_ascii_hexdigit()) {
        return Err("release asset SHA-256 digest is invalid".into());
    }
    let downloads = root.join("downloads").join(version);
    fs::create_dir_all(&downloads).map_err(|error| error.to_string())?;
    let destination = downloads.join(name);
    if destination.is_file() && verify_sha256(&destination, expected_sha256)? {
        let bytes = fs::metadata(&destination).map_err(|error| error.to_string())?.len();
        if asset.byte_size > 0 && bytes != asset.byte_size {
            return Err("cached release download size mismatch".into());
        }
        report_progress(bytes, asset.byte_size);
        return Ok(destination);
    }
    let temporary = destination.with_extension("download.partial");
    if temporary.exists() {
        fs::remove_file(&temporary).map_err(|error| error.to_string())?;
    }
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(60 * 60))
        .user_agent(format!("MFQ-Studio/{}", env!("CARGO_PKG_VERSION")))
        .build()
        .map_err(|error| error.to_string())?;
    let mut response = client
        .get(&asset.download_url)
        .send()
        .and_then(reqwest::blocking::Response::error_for_status)
        .map_err(|error| format!("release download failed: {error}"))?;
    let mut output = File::create(&temporary).map_err(|error| error.to_string())?;
    let mut digest = Sha256::new();
    let mut total = 0_u64;
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let count = response
            .read(&mut buffer)
            .map_err(|error| error.to_string())?;
        if count == 0 {
            break;
        }
        output
            .write_all(&buffer[..count])
            .map_err(|error| error.to_string())?;
        digest.update(&buffer[..count]);
        total = total.saturating_add(count as u64);
        report_progress(total, asset.byte_size);
    }
    output.sync_all().map_err(|error| error.to_string())?;
    if asset.byte_size > 0 && total != asset.byte_size {
        let _ = fs::remove_file(&temporary);
        return Err(format!(
            "release download size mismatch: expected {}, received {total}",
            asset.byte_size
        ));
    }
    let actual = format!("{:x}", digest.finalize());
    if actual != expected_sha256 {
        let _ = fs::remove_file(&temporary);
        return Err("release SHA-256 verification failed".into());
    }
    if destination.exists() {
        fs::remove_file(&destination).map_err(|error| error.to_string())?;
    }
    fs::rename(temporary, &destination).map_err(|error| error.to_string())?;
    Ok(destination)
}

fn verify_sha256(path: &Path, expected: &str) -> Result<bool, String> {
    let mut file = File::open(path).map_err(|error| error.to_string())?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let count = file.read(&mut buffer).map_err(|error| error.to_string())?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()) == expected)
}

#[cfg(any(target_os = "macos", target_os = "windows"))]
fn local_server_is_mfq(port: u16) -> bool {
    let Ok(client) = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(2))
        .no_proxy()
        .build()
    else {
        return false;
    };
    let Ok(response) = client
        .get(format!("http://127.0.0.1:{port}/health"))
        .send()
        .and_then(reqwest::blocking::Response::error_for_status)
    else {
        return false;
    };
    response
        .json::<serde_json::Value>()
        .ok()
        .and_then(|value| {
            value
                .get("service")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned)
        })
        .as_deref()
        == Some("mfq-server")
}

#[cfg(target_os = "macos")]
fn managed_server_process_matches(pid: u32, data_dir: &Path) -> bool {
    let Ok(output) = Command::new("/bin/ps")
        .args(["-p", &pid.to_string(), "-o", "command="])
        .output()
    else {
        return false;
    };
    if !output.status.success() {
        return false;
    }
    let command = String::from_utf8_lossy(&output.stdout);
    command.contains("serve")
        && (command.contains("mfq") || command.contains("python"))
        && command.contains(&data_dir.to_string_lossy().to_string())
}

#[cfg(target_os = "macos")]
fn process_is_running(pid: u32) -> bool {
    Command::new("/bin/kill")
        .args(["-0", &pid.to_string()])
        .status()
        .is_ok_and(|status| status.success())
}

#[cfg(target_os = "windows")]
fn process_is_running(pid: u32) -> bool {
    Command::new("tasklist.exe")
        .args(["/FI", &format!("PID eq {pid}"), "/FO", "CSV", "/NH"])
        .output()
        .is_ok_and(|output| {
            output.status.success()
                && String::from_utf8_lossy(&output.stdout).contains(&pid.to_string())
        })
}

#[cfg(any(target_os = "macos", target_os = "windows"))]
fn stop_managed_server(data_dir: &Path) -> Result<(), String> {
    let Some(config) = read_json::<StudioConnectionConfig>(&data_dir.join(STUDIO_CONFIG_FILE))
    else {
        return Ok(());
    };
    if !local_server_is_mfq(config.local_service_port) {
        return Ok(());
    }
    let pid_path = data_dir.join(SERVER_PID_FILE);
    let Some(pid) = fs::read_to_string(&pid_path)
        .ok()
        .and_then(|value| value.trim().parse::<u32>().ok())
    else {
        return Err("the local MFQ Server is running but its managed PID is unavailable".into());
    };
    #[cfg(target_os = "macos")]
    {
        if !managed_server_process_matches(pid, data_dir) {
            return Err("the saved MFQ Server PID does not match the managed local process".into());
        }
        command_success("/bin/kill", &["-TERM", &pid.to_string()])?;
    }
    #[cfg(target_os = "windows")]
    {
        let status = Command::new("taskkill.exe")
            .args(["/PID", &pid.to_string()])
            .status()
            .map_err(|error| format!("could not stop the managed MFQ Server: {error}"))?;
        if !status.success() {
            return Err(format!("taskkill exited with {status}"));
        }
    }
    for _ in 0..120 {
        if !process_is_running(pid) {
            let _ = fs::remove_file(pid_path);
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(250));
    }
    Err(
        "the managed MFQ Server did not stop within 30 seconds; finish active requests and retry"
            .into(),
    )
}

#[cfg(target_os = "macos")]
fn command_success(program: &str, arguments: &[&str]) -> Result<(), String> {
    let status = Command::new(program)
        .args(arguments)
        .status()
        .map_err(|error| format!("failed to run {program}: {error}"))?;
    status
        .success()
        .then_some(())
        .ok_or_else(|| format!("{program} exited with {status}"))
}

#[cfg(target_os = "macos")]
fn stage_macos_release(
    root: &Path,
    release: &StudioRelease,
    dmg: &Path,
) -> Result<PathBuf, String> {
    let mount = root
        .join("mounts")
        .join(format!("{}-{}", release.version, std::process::id()));
    if mount.exists() {
        fs::remove_dir_all(&mount).map_err(|error| error.to_string())?;
    }
    fs::create_dir_all(&mount).map_err(|error| error.to_string())?;
    let dmg_value = dmg.to_string_lossy().to_string();
    let mount_value = mount.to_string_lossy().to_string();
    if let Err(error) = command_success(
        "/usr/bin/hdiutil",
        &[
            "attach",
            &dmg_value,
            "-nobrowse",
            "-readonly",
            "-mountpoint",
            &mount_value,
        ],
    ) {
        let _ = fs::remove_dir_all(&mount);
        return Err(error);
    }
    let result = (|| {
        let source = fs::read_dir(&mount)
            .map_err(|error| error.to_string())?
            .flatten()
            .map(|entry| entry.path())
            .find(|path| path.extension().and_then(|value| value.to_str()) == Some("app"))
            .ok_or_else(|| "release DMG does not contain an application bundle".to_string())?;
        validate_macos_bundle(&source)?;
        let version_directory = root
            .join("versions")
            .join(safe_version_directory(&release.version)?);
        fs::create_dir_all(&version_directory).map_err(|error| error.to_string())?;
        let destination = version_directory.join("MFQ Studio.app");
        let staging = version_directory.join("MFQ Studio.staging.app");
        if staging.exists() {
            fs::remove_dir_all(&staging).map_err(|error| error.to_string())?;
        }
        let source_value = source.to_string_lossy().to_string();
        let staging_value = staging.to_string_lossy().to_string();
        command_success("/usr/bin/ditto", &[&source_value, &staging_value])?;
        validate_macos_bundle(&staging)?;
        if destination.exists() {
            fs::remove_dir_all(&destination).map_err(|error| error.to_string())?;
        }
        fs::rename(&staging, &destination).map_err(|error| error.to_string())?;
        write_json(
            &version_directory.join(INSTALL_METADATA_FILE),
            &InstalledMetadata {
                release: release.clone(),
                installed_at_epoch_seconds: now_epoch_seconds(),
            },
        )?;
        Ok(destination)
    })();
    let _ = command_success("/usr/bin/hdiutil", &["detach", &mount_value, "-force"]);
    let _ = fs::remove_dir_all(&mount);
    result
}

#[cfg(target_os = "macos")]
fn validate_macos_bundle(bundle: &Path) -> Result<(), String> {
    if bundle.extension().and_then(|value| value.to_str()) != Some("app") {
        return Err("update is not an application bundle".into());
    }
    let plist = bundle.join("Contents/Info.plist");
    let executable_directory = bundle.join("Contents/MacOS");
    if !plist.is_file() || !executable_directory.is_dir() {
        return Err("update application bundle is incomplete".into());
    }
    let output = Command::new("/usr/bin/plutil")
        .args(["-extract", "CFBundleIdentifier", "raw", "-o", "-"])
        .arg(&plist)
        .output()
        .map_err(|error| error.to_string())?;
    if !output.status.success()
        || String::from_utf8_lossy(&output.stdout).trim() != BUNDLE_IDENTIFIER
    {
        return Err("update bundle identifier does not match MFQ Studio".into());
    }
    command_success(
        "/usr/bin/codesign",
        &["--verify", "--deep", "--strict", &bundle.to_string_lossy()],
    )
}

#[cfg(target_os = "macos")]
fn current_bundle() -> Result<PathBuf, String> {
    let executable = std::env::current_exe().map_err(|error| error.to_string())?;
    let macos = executable
        .parent()
        .ok_or_else(|| "current executable has no parent".to_string())?;
    let contents = macos
        .parent()
        .ok_or_else(|| "current executable is not in an app bundle".to_string())?;
    let bundle = contents
        .parent()
        .ok_or_else(|| "current executable is not in an app bundle".to_string())?;
    validate_macos_bundle(bundle)?;
    Ok(bundle.to_path_buf())
}

#[cfg(target_os = "macos")]
fn ensure_bundle_parent_writable(bundle: &Path) -> Result<(), String> {
    let parent = bundle
        .parent()
        .ok_or_else(|| "installed application has no parent directory".to_string())?;
    let probe = parent.join(format!(".mfq-update-write-test-{}", std::process::id()));
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&probe)
        .map_err(|error| {
            format!(
                "MFQ Studio cannot update in its current location ({error}); move it to a writable Applications folder"
            )
        })?;
    fs::remove_file(probe).map_err(|error| error.to_string())
}

#[cfg(target_os = "macos")]
fn snapshot_current_bundle(root: &Path, current: &Path) -> Result<(), String> {
    let version = env!("MFQ_STUDIO_VERSION");
    let directory = root.join("versions").join(safe_version_directory(version)?);
    let destination = directory.join("MFQ Studio.app");
    fs::create_dir_all(&directory).map_err(|error| error.to_string())?;
    if !destination.is_dir() {
        let staging = directory.join("MFQ Studio.current.staging.app");
        if staging.exists() {
            fs::remove_dir_all(&staging).map_err(|error| error.to_string())?;
        }
        command_success(
            "/usr/bin/ditto",
            &[&current.to_string_lossy(), &staging.to_string_lossy()],
        )?;
        validate_macos_bundle(&staging)?;
        fs::rename(staging, &destination).map_err(|error| error.to_string())?;
    }
    validate_macos_bundle(&destination)?;
    let release = StudioRelease {
        version: version.into(),
        tag: format!("v{version}"),
        name: format!("MFQ Studio {version}"),
        notes: "Snapshot retained before an application update.".into(),
        published_at: None,
        page_url: RELEASES_PAGE.into(),
        prerelease: env!("MFQ_STUDIO_RELEASE") != "true",
        asset: Some(StudioReleaseAsset {
            name: "local-snapshot".into(),
            byte_size: 0,
            sha256: None,
            download_url: RELEASES_PAGE.into(),
        }),
    };
    write_json(
        &directory.join(INSTALL_METADATA_FILE),
        &InstalledMetadata {
            release,
            installed_at_epoch_seconds: now_epoch_seconds(),
        },
    )
}

#[cfg(target_os = "macos")]
fn launch_update_helper(root: &Path, source: &Path, target: &Path) -> Result<(), String> {
    let current_executable = std::env::current_exe().map_err(|error| error.to_string())?;
    let helpers = root.join("helpers");
    fs::create_dir_all(&helpers).map_err(|error| error.to_string())?;
    if let Ok(entries) = fs::read_dir(&helpers) {
        for entry in entries.flatten() {
            let _ = fs::remove_file(entry.path());
        }
    }
    let helper = helpers.join(format!("mfq-update-helper-{}", std::process::id()));
    fs::copy(&current_executable, &helper).map_err(|error| error.to_string())?;
    fs::set_permissions(
        &helper,
        fs::metadata(&current_executable)
            .map_err(|error| error.to_string())?
            .permissions(),
    )
    .map_err(|error| error.to_string())?;
    let log = File::create(root.join("updater.log")).map_err(|error| error.to_string())?;
    let error_log = log.try_clone().map_err(|error| error.to_string())?;
    let mut command = Command::new(helper);
    command
        .arg(HELPER_ARGUMENT)
        .arg(std::process::id().to_string())
        .arg(source)
        .arg(target)
        .stdout(Stdio::from(log))
        .stderr(Stdio::from(error_log));
    use std::os::unix::process::CommandExt;
    command.process_group(0);
    command
        .spawn()
        .map(|_| ())
        .map_err(|error| format!("could not start update helper: {error}"))
}

#[cfg(target_os = "macos")]
fn replace_macos_bundle(source: &Path, target: &Path) -> Result<(), String> {
    validate_macos_bundle(source)?;
    validate_macos_bundle(target)?;
    let parent = target
        .parent()
        .ok_or_else(|| "installed application has no parent directory".to_string())?;
    let staging = parent.join(".MFQ Studio.update.app");
    let backup = parent.join(".MFQ Studio.rollback.app");
    if staging.exists() {
        fs::remove_dir_all(&staging).map_err(|error| error.to_string())?;
    }
    command_success(
        "/usr/bin/ditto",
        &[&source.to_string_lossy(), &staging.to_string_lossy()],
    )?;
    validate_macos_bundle(&staging)?;
    if backup.exists() {
        fs::remove_dir_all(&backup).map_err(|error| error.to_string())?;
    }
    fs::rename(target, &backup).map_err(|error| error.to_string())?;
    if let Err(error) = fs::rename(&staging, target) {
        let _ = fs::rename(&backup, target);
        return Err(format!("could not activate staged update: {error}"));
    }
    let _ = fs::remove_dir_all(&backup);
    Command::new("/usr/bin/open")
        .arg(target)
        .spawn()
        .map(|_| ())
        .map_err(|error| format!("updated app was installed but could not be reopened: {error}"))
}

#[cfg(target_os = "macos")]
fn wait_for_parent(pid: u32) {
    for _ in 0..480 {
        let running = Command::new("/bin/kill")
            .args(["-0", &pid.to_string()])
            .status()
            .is_ok_and(|status| status.success());
        if !running {
            return;
        }
        std::thread::sleep(Duration::from_millis(250));
    }
}

pub fn maybe_run_update_helper() -> bool {
    let arguments: Vec<String> = std::env::args().collect();
    if arguments.get(1).map(String::as_str) != Some(HELPER_ARGUMENT) {
        return false;
    }
    #[cfg(target_os = "macos")]
    {
        let result = (|| {
            let pid = arguments
                .get(2)
                .ok_or_else(|| "update helper is missing its parent pid".to_string())?
                .parse::<u32>()
                .map_err(|error| format!("invalid update helper parent pid: {error}"))?;
            let source = PathBuf::from(
                arguments
                    .get(3)
                    .ok_or_else(|| "update helper is missing its source bundle".to_string())?,
            );
            let target = PathBuf::from(
                arguments
                    .get(4)
                    .ok_or_else(|| "update helper is missing its target bundle".to_string())?,
            );
            wait_for_parent(pid);
            replace_macos_bundle(&source, &target)
        })();
        if let Err(error) = result {
            eprintln!("MFQ Studio update helper failed: {error}");
            if let Some(target) = arguments.get(4) {
                let _ = Command::new("/usr/bin/open").arg(target).spawn();
            }
        }
        if let Ok(helper) = std::env::current_exe() {
            let _ = fs::remove_file(helper);
        }
    }
    true
}

fn set_progress(
    progress: &Arc<StdMutex<Option<UpdateProgress>>>,
    tag: &str,
    stage: &str,
    received_bytes: u64,
    total_bytes: u64,
) {
    if let Ok(mut value) = progress.lock() {
        *value = Some(UpdateProgress {
            tag: tag.to_string(),
            stage: stage.to_string(),
            received_bytes,
            total_bytes,
        });
    }
}

#[tauri::command]
pub fn studio_update_progress(state: State<'_, UpdateState>) -> Option<UpdateProgress> {
    state.progress.lock().ok().and_then(|value| value.clone())
}

#[tauri::command]
pub async fn studio_update_status(
    app: AppHandle,
    state: State<'_, UpdateState>,
    force: bool,
) -> Result<UpdateStatus, String> {
    let _guard = state.lock.lock().await;
    let root = update_root(&app)?;
    tokio::task::spawn_blocking(move || update_status(&root, force))
        .await
        .map_err(|error| error.to_string())
}

#[tauri::command]
pub async fn studio_update_set_automatic(
    app: AppHandle,
    state: State<'_, UpdateState>,
    enabled: bool,
) -> Result<UpdateStatus, String> {
    let _guard = state.lock.lock().await;
    let root = update_root(&app)?;
    tokio::task::spawn_blocking(move || {
        write_json(
            &root.join(PREFERENCES_FILE),
            &UpdatePreferences {
                automatic_check: enabled,
                ..preferences(&root)
            },
        )?;
        Ok(update_status(&root, false))
    })
    .await
    .map_err(|error| error.to_string())?
}

#[tauri::command]
pub async fn studio_update_set_download(
    app: AppHandle,
    state: State<'_, UpdateState>,
    enabled: bool,
) -> Result<UpdateStatus, String> {
    let _guard = state.lock.lock().await;
    let root = update_root(&app)?;
    tokio::task::spawn_blocking(move || {
        write_json(&root.join(PREFERENCES_FILE), &UpdatePreferences {
            automatic_download: enabled, ..preferences(&root)
        })?;
        Ok(update_status(&root, false))
    }).await.map_err(|error| error.to_string())?
}

#[tauri::command]
pub async fn studio_update_download(
    app: AppHandle,
    state: State<'_, UpdateState>,
    tag: String,
) -> Result<InstalledVersion, String> {
    let _guard = state.lock.lock().await;
    let root = update_root(&app)?;
    let progress = Arc::clone(&state.progress);
    let progress_for_task = Arc::clone(&progress);
    let result = tokio::task::spawn_blocking(move || {
        let cache: ReleasesCache = read_json(&root.join(RELEASES_FILE))
            .ok_or_else(|| "check for updates before downloading a release".to_string())?;
        let release = cache
            .releases
            .into_iter()
            .find(|value| value.tag == tag)
            .ok_or_else(|| "release is not present in the verified GitHub catalog".to_string())?;
        if release.prerelease {
            return Err("experimental versions cannot be downloaded by the Release updater".into());
        }
        let byte_size = release.asset.as_ref().ok_or_else(|| "this release has no verified installer for this platform".to_string())?.byte_size;
        set_progress(
            &progress_for_task,
            &release.tag,
            "downloading",
            0,
            byte_size,
        );
        let progress_for_download = Arc::clone(&progress_for_task);
        let progress_tag = release.tag.clone();
        let asset = download_asset(&root, &release, move |received, total| {
            set_progress(
                &progress_for_download,
                &progress_tag,
                "downloading",
                received,
                total,
            );
        })?;
        set_progress(
            &progress_for_task,
            &release.tag,
            "preparing",
            byte_size,
            byte_size,
        );
        #[cfg(target_os = "macos")]
        stage_macos_release(&root, &release, &asset)?;
        #[cfg(target_os = "windows")]
        {
            let directory = root
                .join("versions")
                .join(safe_version_directory(&release.version)?);
            fs::create_dir_all(&directory).map_err(|error| error.to_string())?;
            let destination = directory.join(&release.asset.as_ref().unwrap().name);
            fs::copy(&asset, destination).map_err(|error| error.to_string())?;
            write_json(
                &directory.join(INSTALL_METADATA_FILE),
                &InstalledMetadata {
                    release: release.clone(),
                    installed_at_epoch_seconds: now_epoch_seconds(),
                },
            )?;
        }
        let _ = fs::remove_dir_all(root.join("downloads").join(&release.version));
        installed_versions(&root)
            .into_iter()
            .find(|value| value.version == release.version)
            .ok_or_else(|| "downloaded release was not staged".to_string())
    })
    .await
    .map_err(|error| error.to_string());
    if let Ok(mut value) = progress.lock() {
        *value = None;
    }
    result?
}

#[tauri::command]
pub async fn studio_update_install(
    app: AppHandle,
    state: State<'_, UpdateState>,
    version: String,
) -> Result<(), String> {
    let _guard = state.lock.lock().await;
    let root = update_root(&app)?;
    let data_dir = root
        .parent()
        .ok_or_else(|| "update directory has no application data parent".to_string())?
        .to_path_buf();
    let version = safe_version_directory(&version)?;
    #[cfg(target_os = "macos")]
    {
        tokio::task::spawn_blocking(move || {
            let source = cached_installable(&root.join("versions").join(version))
                .ok_or_else(|| "selected version has not been downloaded".to_string())?;
            let target = current_bundle()?;
            ensure_bundle_parent_writable(&target)?;
            snapshot_current_bundle(&root, &target)?;
            stop_managed_server(&data_dir)?;
            launch_update_helper(&root, &source, &target)
        })
        .await
        .map_err(|error| error.to_string())??;
        app.exit(0);
        return Ok(());
    }
    #[cfg(target_os = "windows")]
    {
        tokio::task::spawn_blocking(move || {
            let installer = cached_installable(&root.join("versions").join(version))
                .ok_or_else(|| "selected version has not been downloaded".to_string())?;
            stop_managed_server(&data_dir)?;
            Command::new(installer)
                .spawn()
                .map(|_| ())
                .map_err(|error| format!("could not start the installer: {error}"))
        })
        .await
        .map_err(|error| error.to_string())??;
        app.exit(0);
        return Ok(());
    }
    #[allow(unreachable_code)]
    Err("application updates are not available on this platform".into())
}

#[tauri::command]
pub async fn studio_update_delete(
    app: AppHandle,
    state: State<'_, UpdateState>,
    version: String,
) -> Result<UpdateStatus, String> {
    let _guard = state.lock.lock().await;
    if version == env!("MFQ_STUDIO_VERSION") {
        return Err("the running version cannot be removed".into());
    }
    let root = update_root(&app)?;
    let version = safe_version_directory(&version)?;
    tokio::task::spawn_blocking(move || {
        let directory = root.join("versions").join(&version);
        if directory.exists() {
            fs::remove_dir_all(directory).map_err(|error| error.to_string())?;
        }
        let downloads = root.join("downloads").join(&version);
        if downloads.exists() {
            fs::remove_dir_all(downloads).map_err(|error| error.to_string())?;
        }
        Ok(update_status(&root, false))
    })
    .await
    .map_err(|error| error.to_string())?
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new() -> Self {
            let directory = std::env::temp_dir().join(format!("mfq-updater-{}-{}",
                std::process::id(), SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()));
            fs::create_dir(&directory).unwrap();
            Self(directory)
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); }
    }

    fn fixture_release() -> StudioRelease {
        StudioRelease { version: "0.3.2".into(), tag: "v0.3.2".into(), name: "MFQ".into(),
            notes: String::new(), published_at: None, page_url: RELEASES_PAGE.into(), prerelease: false,
            asset: Some(StudioReleaseAsset { name: "MFQ.dmg".into(), byte_size: 3,
                sha256: Some(format!("{:x}", Sha256::digest(b"abc"))),
                download_url: "https://github.com/Tylogi/MFQ/releases/download/v0.3.2/MFQ.dmg".into() }) }
    }

    #[test]
    fn corrupted_release_cache_is_rejected_before_any_download_or_directory_creation() {
        let root = TestDirectory::new();
        for version in ["../outside", "../../outside", "/outside", "v0.3.2", "0.3.2/extra", "0.3.2+dev.1234"] {
            let mut release = fixture_release();
            release.version = version.into();
            assert!(download_asset(&root.0, &release, |_, _| {}).is_err());
        }
        for name in ["../outside", "..\\outside", "C:outside"] {
            let mut release = fixture_release();
            release.asset.as_mut().unwrap().name = name.into();
            assert!(download_asset(&root.0, &release, |_, _| {}).is_err());
        }
        let mut release = fixture_release();
        release.asset.as_mut().unwrap().sha256 = Some("invalid".into());
        assert!(download_asset(&root.0, &release, |_, _| {}).is_err());
        assert_eq!(fs::read_dir(&root.0).unwrap().count(), 0);
    }

    #[test]
    fn verified_cached_download_is_reused_without_network_or_rewriting() {
        let root = TestDirectory::new();
        let release = fixture_release();
        let asset = release.asset.as_ref().unwrap();
        let directory = root.0.join("downloads").join(&release.version);
        fs::create_dir_all(&directory).unwrap();
        let path = directory.join(&asset.name);
        fs::write(&path, b"abc").unwrap();
        let mut progress = Vec::new();
        assert_eq!(download_asset(&root.0, &release, |bytes, total| progress.push((bytes, total))).unwrap(), path);
        assert_eq!(progress, vec![(3, 3)]);
        assert_eq!(fs::read(&path).unwrap(), b"abc");
        let mut corrupt = release.clone();
        corrupt.asset.as_mut().unwrap().byte_size = 4;
        assert!(download_asset(&root.0, &corrupt, |_, _| {}).unwrap_err().contains("size mismatch"));
        assert_eq!(fs::read(&path).unwrap(), b"abc");
        assert!(!verify_sha256(&path, &"0".repeat(64)).unwrap());
    }

    #[test]
    fn extracts_versions_from_current_and_legacy_release_tags() {
        assert_eq!(extract_version("v0.3.2").as_deref(), Some("0.3.2"));
        assert_eq!(
            extract_version("studio-v0.1.0-multimodal-20260828").as_deref(),
            Some("0.1.0")
        );
        assert!(compare_versions("0.3.2", "0.3.1").is_gt());
    }

    #[test]
    fn accepts_only_repository_release_asset_urls() {
        assert!(validate_download_url(
            "https://github.com/Tylogi/TyloQuant/releases/download/v0.3.2/MFQ.dmg"
        )
        .is_ok());
        assert!(validate_download_url(
            "https://github.com/Tylogi/MFQ/releases/download/v0.3.2/MFQ.dmg"
        )
        .is_ok());
        assert!(validate_download_url("https://example.com/MFQ.dmg").is_err());
        assert!(validate_download_url(
            "https://user:password@github.com/Tylogi/MFQ/releases/download/v0.3.2/MFQ.dmg"
        )
        .is_err());
        assert!(validate_download_url(
            "https://github.com/another/project/releases/download/v1/MFQ.dmg"
        )
        .is_err());
        assert_eq!(
            safe_asset_name("MFQ.Studio_0.3.2_aarch64.dmg").unwrap(),
            "MFQ.Studio_0.3.2_aarch64.dmg"
        );
        assert!(safe_asset_name("../MFQ.dmg").is_err());
    }

    #[test]
    fn selects_the_native_macos_asset() {
        let release = GitHubRelease {
            tag_name: "v0.3.2".into(),
            name: None,
            body: None,
            html_url: RELEASES_PAGE.into(),
            published_at: None,
            draft: false,
            prerelease: false,
            assets: vec![GitHubAsset {
                name: "MFQ.Studio_0.3.2_aarch64.dmg".into(),
                browser_download_url:
                    "https://github.com/Tylogi/TyloQuant/releases/download/v0.3.2/MFQ.dmg".into(),
                size: 42,
                content_type: "application/x-apple-diskimage".into(),
                digest: Some(format!("sha256:{}", "a".repeat(64))),
            }],
        };
        let mut unsigned = release.clone();
        unsigned.assets[0].digest = None;
        assert!(normalized_release(unsigned).unwrap().asset.is_none());
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        assert_eq!(normalized_release(release).unwrap().version, "0.3.2");
    }

    #[test]
    fn release_channel_rejects_experimental_tags_and_drafts() {
        for tag in ["v0.4.0rc1", "v0.4.0a1", "v0.4.0b1", "v0.4.0-dev.1", "nightly", "0.4.0+test"] {
            assert!(!is_release_tag(tag), "{tag}");
        }
        assert!(is_release_tag("v0.4.0"));
        assert!(is_release_tag("studio-v0.1.0-multimodal-20260828"));
        let mut release = GitHubRelease { tag_name: "v0.4.0".into(), name: None, body: Some("notes".into()),
            html_url: RELEASES_PAGE.into(), published_at: None, draft: true, prerelease: false, assets: vec![] };
        assert!(normalized_release(release.clone()).is_none());
        release.draft = false;
        release.prerelease = true;
        assert!(normalized_release(release.clone()).is_none());
        release.prerelease = false;
        let visible = normalized_release(release).unwrap();
        assert_eq!(visible.notes, "notes");
        assert!(visible.asset.is_none());
    }

    #[test]
    fn migrates_preferences_and_keeps_development_snapshots_separate() {
        let preferences: UpdatePreferences = serde_json::from_str(r#"{"automatic_check":false}"#).unwrap();
        assert!(!preferences.automatic_download);
        assert_eq!(safe_version_directory("0.3.2+dev.1234abcd.modified").unwrap(), "0.3.2+dev.1234abcd.modified");
        assert!(safe_version_directory("0.3.2+dev.../elsewhere").is_err());
        assert!(safe_version_directory("../0.3.2").is_err());
        assert!(safe_version_directory("0.3.2+dev.").is_err());
        assert!(compare_versions("0.3.3", "0.3.2+dev.1234abcd.modified").is_gt());
    }
}
