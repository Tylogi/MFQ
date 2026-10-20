#![cfg_attr(all(target_os = "windows", not(debug_assertions)), windows_subsystem = "windows")]

mod updater;
#[cfg(target_os = "windows")]
mod windows_icon;

use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::time::Duration;
use tauri::{AppHandle, Manager, State};
use tokio::sync::Mutex;
use url::Url;

const CONFIG_FILE: &str = "studio.json";
const LOG_FILE: &str = "mfq-server.log";
const LOG_ROTATE_BYTES: u64 = 16 * 1024 * 1024;
const CREDENTIAL_SERVICE: &str = "MFQ Studio";
const CREDENTIAL_ACCOUNT: &str = "mfq-server-api-key";

#[tauri::command]
fn studio_custom_titlebar() -> bool {
    cfg!(target_os = "windows")
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum RuntimeMode {
    Local,
    Remote,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
struct StudioConfig {
    mode: RuntimeMode,
    remote_url: String,
    local_service_port: u16,
}

impl Default for StudioConfig {
    fn default() -> Self {
        Self {
            mode: RuntimeMode::Local,
            remote_url: "http://127.0.0.1:8090".into(),
            local_service_port: 8090,
        }
    }
}

#[derive(Debug, Serialize)]
struct StudioStatus {
    config: StudioConfig,
    service_url: String,
    reachable: bool,
    managed_pid: Option<u32>,
    connection: &'static str,
    server: Option<ServerIdentity>,
    connection_error: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(default)]
struct ServerIdentity {
    service: String,
    protocol_version: String,
    studio_protocol_version: String,
    version: String,
    instance_id: String,
    workspace_id: String,
    pid: u32,
    launch_mode: String,
}

impl ServerIdentity {
    fn validate(&self) -> Result<(), String> {
        if self.service != "mfq-server" || self.protocol_version != "1.0"
            || self.studio_protocol_version != "1.0" || self.version != env!("MFQ_SERVER_VERSION")
        {
            return Err(format!("Incompatible MFQ Server (version {}). Update the server or stop it so Studio can start its bundled backend.",
                if self.version.is_empty() { "unknown" } else { &self.version }));
        }
        if uuid::Uuid::parse_str(&self.instance_id).is_err() || self.pid == 0
            || self.workspace_id.len() != 64 || !self.workspace_id.bytes().all(|byte| byte.is_ascii_hexdigit())
            || !matches!(self.launch_mode.as_str(), "cli" | "studio")
        {
            return Err("MFQ Server returned an invalid instance identity. Update or restart the server.".into());
        }
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
struct RegisteredModel {
    name: String,
}

#[derive(Debug, Deserialize)]
struct RegisteredModelList {
    data: Vec<RegisteredModel>,
}

#[derive(Serialize)]
struct SelectedModelDirectory {
    path: String,
    names: Vec<String>,
}

#[derive(Default)]
struct StudioState {
    start_lock: Mutex<()>,
    server_input: std::sync::Mutex<Option<(u32, String, ChildStdin)>>,
    server_child: std::sync::Mutex<Option<Child>>,
    external_server: std::sync::Mutex<Option<(String, ServerIdentity)>>,
}

impl StudioState {
    fn reap_exited_server(&self) {
        if let Ok(mut child) = self.server_child.lock() {
            if let Some(process) = child.as_mut() {
                if process.try_wait().ok().flatten().is_some() {
                    self.release_server(process.id());
                    child.take();
                }
            }
        }
    }

    fn own_server(&self, child: &mut Child, instance_id: &str) -> Result<(), String> {
        let input = child.stdin.take().ok_or("MFQ Server lifecycle pipe is unavailable")?;
        let mut owned = self.server_input.lock().map_err(|error| error.to_string())?;
        if owned.is_some() { return Err("Studio already owns a server. Stop it before starting another one.".into()); }
        *owned = Some((child.id(), instance_id.to_string(), input));
        Ok(())
    }

    fn release_server(&self, pid: u32) {
        if let Ok(mut input) = self.server_input.lock() {
            if input.as_ref().is_some_and(|(owner, _, _)| *owner == pid) {
                input.take();
            }
        }
    }

    fn stop_owned_server(&self) -> Result<(), String> {
        let mut child = self.server_child.lock().map_err(|error| error.to_string())?;
        let Some(process) = child.as_mut() else { return Ok(()); };
        self.release_server(process.id());
        for _ in 0..120 {
            if process.try_wait().map_err(|error| format!("could not wait for MFQ Server: {error}"))?.is_some() {
                child.take();
                return Ok(());
            }
            std::thread::sleep(Duration::from_millis(250));
        }
        Err("MFQ Server did not stop within 30 seconds; finish active requests and retry".into())
    }

    fn owns(&self, server: &ServerIdentity) -> bool {
        self.server_input.lock().ok().is_some_and(|input| input.as_ref().is_some_and(
            |(pid, instance, _)| *pid == server.pid && *instance == server.instance_id && server.launch_mode == "studio"))
    }

    fn accepts(&self, url: &str, server: &ServerIdentity) -> bool {
        self.external_server.lock().ok().is_some_and(|accepted| accepted.as_ref().is_some_and(
            |(endpoint, identity)| endpoint == url && identity == server))
    }
}

fn app_data_dir(app: &AppHandle) -> Result<PathBuf, String> {
    let path = app
        .path()
        .app_local_data_dir()
        .map_err(|error| error.to_string())?;
    fs::create_dir_all(&path).map_err(|error| error.to_string())?;
    Ok(path)
}

fn normalized_http_url(value: &str, field: &str) -> Result<String, String> {
    let trimmed = value.trim().trim_end_matches('/');
    let parsed = Url::parse(trimmed).map_err(|error| format!("invalid {field}: {error}"))?;
    if !matches!(parsed.scheme(), "http" | "https") || parsed.host_str().is_none() {
        return Err(format!("{field} must be an HTTP or HTTPS URL"));
    }
    if !parsed.username().is_empty() || parsed.password().is_some() {
        return Err(format!("{field} must not contain credentials"));
    }
    Ok(trimmed.to_string())
}

fn validate_config(mut config: StudioConfig) -> Result<StudioConfig, String> {
    config.remote_url = normalized_http_url(&config.remote_url, "remote_url")?;
    if config.local_service_port == 0 {
        return Err("local_service_port must be greater than zero".into());
    }
    Ok(config)
}

fn config_path(app: &AppHandle) -> Result<PathBuf, String> {
    Ok(app_data_dir(app)?.join(CONFIG_FILE))
}

fn load_config(app: &AppHandle) -> Result<StudioConfig, String> {
    let path = config_path(app)?;
    if !path.exists() {
        return Ok(StudioConfig::default());
    }
    let bytes = fs::read(path).map_err(|error| error.to_string())?;
    let config = serde_json::from_slice(&bytes).map_err(|error| error.to_string())?;
    validate_config(config)
}

fn save_config(app: &AppHandle, config: &StudioConfig) -> Result<(), String> {
    let path = config_path(app)?;
    let bytes = serde_json::to_vec_pretty(config).map_err(|error| error.to_string())?;
    fs::write(path, bytes).map_err(|error| error.to_string())
}

fn local_service_url(config: &StudioConfig) -> String {
    format!("http://127.0.0.1:{}", config.local_service_port)
}

fn service_url(config: &StudioConfig) -> String {
    match config.mode {
        RuntimeMode::Local => local_service_url(config),
        RuntimeMode::Remote => config.remote_url.clone(),
    }
}

fn is_loopback_service(base_url: &str) -> bool {
    let Ok(url) = Url::parse(base_url) else { return false; };
    match url.host() {
        Some(url::Host::Domain(host)) => host.eq_ignore_ascii_case("localhost"),
        Some(url::Host::Ipv4(address)) => address.is_loopback(),
        Some(url::Host::Ipv6(address)) => address.is_loopback(),
        None => false,
    }
}

fn server_client(base_url: &str, timeout: Duration) -> Result<reqwest::Client, String> {
    let mut builder = reqwest::Client::builder().timeout(timeout);
    if is_loopback_service(base_url) {
        builder = builder.no_proxy();
    }
    builder.build().map_err(|error| error.to_string())
}

fn check_local_ports(data_dir: &Path, openai_port: u16) -> Result<(), String> {
    let anthropic_port = fs::read(data_dir.join("listener.json")).ok()
        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
        .and_then(|value| value.get("anthropic_port").and_then(|port| port.as_u64()))
        .and_then(|port| u16::try_from(port).ok())
        .filter(|port| *port > 0)
        .unwrap_or(8091);
    if openai_port == anthropic_port {
        return Err(format!("OpenAI and Anthropic cannot share port {openai_port}. Choose different API ports."));
    }
    for (protocol, port) in [("OpenAI", openai_port), ("Anthropic", anthropic_port)] {
        std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, port))
            .map_err(|error| match error.kind() {
                std::io::ErrorKind::AddrInUse => format!("The {protocol} API port {port} is already in use. Close the other service or choose a different port."),
                std::io::ErrorKind::PermissionDenied => format!("Permission denied for the {protocol} API port {port}. Choose a different port."),
                _ => format!("Could not use the {protocol} API port {port}: {error}"),
            })?;
    }
    Ok(())
}

fn startup_failure(log_path: &Path, offset: u64, summary: &str) -> String {
    let detail = (|| -> std::io::Result<String> {
        let mut log = File::open(log_path)?;
        let end = log.metadata()?.len();
        log.seek(SeekFrom::Start(offset.max(end.saturating_sub(8192))))?;
        let mut bytes = Vec::new();
        log.take(8192).read_to_end(&mut bytes)?;
        Ok(String::from_utf8_lossy(&bytes).trim().to_string())
    })().unwrap_or_default();
    if detail.is_empty() {
        format!("{summary}\nServer log: {}", log_path.display())
    } else {
        format!("{summary}\n\n{detail}\n\nServer log: {}", log_path.display())
    }
}

async fn is_mfq_server(client: &reqwest::Client, base_url: &str) -> bool {
    let response = match client.get(format!("{base_url}/health")).send().await {
        Ok(response) if response.status().is_success() => response,
        _ => return false,
    };
    let payload = match response.json::<serde_json::Value>().await {
        Ok(payload) => payload,
        Err(_) => return false,
    };
    payload.get("service").and_then(|value| value.as_str()) == Some("mfq-server")
}

async fn probe_server(client: &reqwest::Client, base_url: &str) -> Result<Option<ServerIdentity>, String> {
    if !is_mfq_server(client, base_url).await { return Ok(None); }
    let response = client.get(format!("{base_url}/health/identity")).send().await
        .map_err(|_| "Could not verify the MFQ Server instance. Restart or update the server.".to_string())?;
    if !response.status().is_success() {
        return Err("This MFQ Server does not support Studio's instance handshake. Update it or stop it so Studio can start its bundled backend.".into());
    }
    let server: ServerIdentity = response.json().await
        .map_err(|_| "MFQ Server returned an invalid instance identity.".to_string())?;
    server.validate()?;
    Ok(Some(server))
}

fn with_server_credential(request: reqwest::RequestBuilder, token: &str) -> reqwest::RequestBuilder {
    let token = token.trim();
    if token.is_empty() { request } else { request.bearer_auth(token) }
}

fn open_log(path: &Path) -> Result<File, String> {
    if path
        .metadata()
        .map(|metadata| metadata.len() >= LOG_ROTATE_BYTES)
        .unwrap_or(false)
    {
        let backup = path.with_extension("log.1");
        if backup.exists() {
            fs::remove_file(&backup).map_err(|error| error.to_string())?;
        }
        fs::rename(path, backup).map_err(|error| error.to_string())?;
    }
    OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|error| error.to_string())
}

fn configure_detached(command: &mut Command) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        command.creation_flags(CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW);
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
}

fn executable_on_path(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    #[cfg(windows)]
    let names = [
        format!("{name}.exe"),
        format!("{name}.cmd"),
        name.to_string(),
    ];
    #[cfg(not(windows))]
    let names = [name.to_string()];
    for directory in std::env::split_paths(&path) {
        for candidate in &names {
            let path = directory.join(candidate);
            if path.is_file() {
                return Some(path);
            }
        }
    }
    None
}

fn packaged_executable(app: &AppHandle, names: &[&str]) -> Option<PathBuf> {
    let mut directories = Vec::new();
    if let Ok(current) = std::env::current_exe() {
        if let Some(parent) = current.parent() {
            directories.push(parent.to_path_buf());
        }
    }
    if let Ok(resource) = app.path().resource_dir() {
        directories.push(resource);
    }
    for directory in directories {
        for name in names {
            let candidate = directory.join(name);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

fn mfq_program(app: &AppHandle) -> Result<(PathBuf, bool), String> {
    #[cfg(windows)]
    let bundled_names = ["mfq.exe", "mfq.cmd", "mfq"];
    #[cfg(not(windows))]
    let bundled_names = ["mfq"];
    if let Ok(resource_dir) = app.path().resource_dir() {
        #[cfg(windows)]
        let bundled_cli = resource_dir.join("mfq-cli").join("mfq-cli.exe");
        #[cfg(not(windows))]
        let bundled_cli = resource_dir.join("mfq-cli").join("mfq-cli");
        if bundled_cli.is_file() {
            return Ok((bundled_cli, false));
        }
    }
    if let Some(path) = packaged_executable(app, &bundled_names) {
        return Ok((path, false));
    }
    if let Some(path) = executable_on_path("mfq") {
        return Ok((path, false));
    }
    let python_name = if cfg!(windows) { "python" } else { "python3" };
    executable_on_path(python_name)
        .map(|path| (path, true))
        .ok_or_else(|| "MFQ is not bundled and neither mfq nor Python is on PATH".into())
}

fn packaged_runtime(app: &AppHandle) -> Option<PathBuf> {
    #[cfg(target_os = "macos")]
    let names = ["mfq-decode-metal"];
    #[cfg(target_os = "windows")]
    let names = ["mfq-runtime.exe"];
    #[cfg(all(unix, not(target_os = "macos")))]
    let names = ["mfq-runtime"];
    packaged_executable(app, &names)
}

fn packaged_resource(app: &AppHandle, name: &str) -> Option<PathBuf> {
    let resource_dir = app.path().resource_dir().ok()?;
    let candidate = resource_dir.join(name);
    candidate.is_file().then_some(candidate)
}

#[cfg(target_os = "macos")]
fn packaged_framework(name: &str) -> Option<PathBuf> {
    let executable = std::env::current_exe().ok()?;
    let contents = executable.parent()?.parent()?;
    let candidate = contents.join("Frameworks").join(name);
    candidate.is_file().then_some(candidate)
}

async fn status_for(app: &AppHandle, config: StudioConfig) -> Result<StudioStatus, String> {
    app.state::<StudioState>().reap_exited_server();
    let service_url = service_url(&config);
    let client = server_client(&service_url, Duration::from_secs(2))?;
    let probe = probe_server(&client, &service_url).await;
    let (server, connection_error) = match probe {
        Ok(server) => (server, None),
        Err(error) => (None, Some(error)),
    };
    let state = app.state::<StudioState>();
    let owned = server.as_ref().is_some_and(|server| state.owns(server));
    let accepted = server.as_ref().is_some_and(|server| state.accepts(&service_url, server));
    let connection = if connection_error.is_some() { "incompatible" }
        else if owned { "managed" }
        else if server.is_some() && accepted { "external" }
        else if server.is_some() { "confirmation_required" }
        else { "offline" };
    let reachable = owned || accepted;
    let managed_pid = server.as_ref().filter(|_| owned).map(|server| server.pid);
    Ok(StudioStatus {
        config,
        service_url,
        reachable,
        managed_pid,
        connection,
        server,
        connection_error,
    })
}

async fn start_local(app: &AppHandle, config: StudioConfig) -> Result<StudioStatus, String> {
    app.state::<StudioState>().reap_exited_server();
    let service_url = local_service_url(&config);
    let client = server_client(&service_url, Duration::from_millis(500))?;
    if let Some(server) = probe_server(&client, &service_url).await? {
        if !app.state::<StudioState>().owns(&server) {
            return Err("An independently managed MFQ Server is already running. Connect to it explicitly, or stop it before starting Studio's backend.".into());
        }
        return status_for(app, config).await;
    }

    let state = app.state::<StudioState>();
    if state.server_input.lock().map_err(|error| error.to_string())?.is_some() {
        return Err("Studio's server is already starting or unavailable. Retry after it stops; a second backend will not be launched.".into());
    }

    let data_dir = app_data_dir(app)?;
    check_local_ports(&data_dir, config.local_service_port)?;
    let (program, python_module) = mfq_program(app)?;
    let log_path = data_dir.join(LOG_FILE);
    let stdout = open_log(&log_path)?;
    let log_offset = stdout.metadata().map_err(|error| error.to_string())?.len();
    let stderr = stdout.try_clone().map_err(|error| error.to_string())?;
    let mut command = Command::new(program);
    let instance_id = uuid::Uuid::new_v4().to_string();
    if python_module {
        command.arg("-m").arg("mfq.cli");
    }
    command
        .arg("serve")
        .arg("--exit-on-stdin-close")
        .arg("--studio-instance-id")
        .arg(&instance_id)
        .arg("--no-access-log")
        .arg("--data-dir")
        .arg(&data_dir)
        .arg("--work-dir")
        .arg(&data_dir)
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(config.local_service_port.to_string());
    if let Some(runtime) = packaged_runtime(app) {
        command.arg("--running-executable").arg(runtime);
    }
    #[cfg(windows)]
    if let Ok(resource_dir) = app.path().resource_dir() {
        // The CUDA, PyTorch, and MSVC DLLs are bundled as resources rather
        // than alongside the signed Studio executable.  Put that directory
        // first in the worker's inherited DLL search path.
        let mut paths = vec![resource_dir];
        if let Some(existing) = std::env::var_os("PATH") {
            paths.extend(std::env::split_paths(&existing));
        }
        if let Ok(path) = std::env::join_paths(paths) {
            command.env("PATH", path);
        }
    }
    if let Some(metallib) = packaged_resource(app, "mlx.metallib") {
        command.env("MFQ_MLX_METALLIB", metallib);
    }
    #[cfg(target_os = "macos")]
    if let Some(video_library) = packaged_framework("libmfq_avfoundation_video.dylib") {
        command.env("MFQ_AVFOUNDATION_VIDEO_LIBRARY", video_library);
    }
    command
        .stdin(Stdio::piped())
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr))
        .current_dir(&data_dir);
    configure_detached(&mut command);
    let mut child = command
        .spawn()
        .map_err(|error| format!("failed to start MFQ Server: {error}"))?;
    state.own_server(&mut child, &instance_id)?;
    let result = async {
        for _ in 0..3600 {
            if let Some(status) = child.try_wait().map_err(|error| error.to_string())? {
                return Err(startup_failure(&log_path, log_offset,
                    &format!("MFQ Server could not start ({status}).")));
            }
            if let Some(server) = probe_server(&client, &service_url).await? {
                if !state.owns(&server) {
                    return Err("Another server appeared on the port during startup. Studio did not connect to it.".into());
                }
                return status_for(app, config).await;
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
        Err(startup_failure(&log_path, log_offset,
            "MFQ Server did not become ready within 30 minutes."))
    }.await;
    if result.is_err() {
        state.release_server(child.id());
    } else {
        *state.server_child.lock().map_err(|error| error.to_string())? = Some(child);
    }
    result
}

#[tauri::command]
async fn studio_status(app: AppHandle) -> Result<StudioStatus, String> {
    status_for(&app, load_config(&app)?).await
}

#[tauri::command]
async fn studio_configure(
    app: AppHandle,
    state: State<'_, StudioState>,
    config: StudioConfig,
) -> Result<StudioStatus, String> {
    let config = validate_config(config)?;
    let _guard = state.start_lock.lock().await;
    let previous = load_config(&app)?;
    if matches!(config.mode, RuntimeMode::Local)
        && previous.local_service_port != config.local_service_port
    {
        let previous_url = local_service_url(&previous);
        let client = server_client(&previous_url, Duration::from_secs(10))?;
        if let Some(server) = probe_server(&client, &previous_url).await? {
            if !state.owns(&server) {
                return Err("This server is managed outside Studio. Change its port through its own service settings.".into());
            }
            let token = studio_credential_get()?;
            let request = client.put(format!("{previous_url}/api/v1/runtime/listener"))
                .header("X-MFQ-Instance-ID", &server.instance_id)
                .json(&serde_json::json!({ "port": config.local_service_port }));
            let response = with_server_credential(request, &token).send().await.map_err(|error| error.to_string())?;
            if !response.status().is_success() {
                return Err(format!("failed to change server port: {}", response.text().await.unwrap_or_default()));
            }
        }
    }
    let status = match config.mode {
        RuntimeMode::Remote => status_for(&app, config.clone()).await?,
        RuntimeMode::Local => {
            let current = status_for(&app, config.clone()).await?;
            if current.connection == "offline" { start_local(&app, config.clone()).await? } else { current }
        }
    };
    save_config(&app, &config)?;
    Ok(status)
}

#[tauri::command]
async fn studio_connect_existing(app: AppHandle, state: State<'_, StudioState>,
    instance_id: String, workspace_id: String) -> Result<StudioStatus, String> {
    let _guard = state.start_lock.lock().await;
    let config = load_config(&app)?;
    let endpoint = service_url(&config);
    let client = server_client(&endpoint, Duration::from_secs(2))?;
    let server = probe_server(&client, &endpoint).await?
        .ok_or("The server stopped before connecting. Refresh and try again.")?;
    if server.instance_id != instance_id || server.workspace_id != workspace_id {
        return Err("The server instance changed. Refresh and confirm the new server before connecting.".into());
    }
    *state.external_server.lock().map_err(|error| error.to_string())? = Some((endpoint, server));
    status_for(&app, config).await
}

#[tauri::command]
async fn studio_start_local(
    app: AppHandle,
    state: State<'_, StudioState>,
) -> Result<StudioStatus, String> {
    let mut config = load_config(&app)?;
    config.mode = RuntimeMode::Local;
    let _guard = state.start_lock.lock().await;
    let status = start_local(&app, config.clone()).await?;
    save_config(&app, &config)?;
    Ok(status)
}

#[tauri::command]
async fn studio_select_model_directory(
    app: AppHandle,
    initial_directory: String,
) -> Result<Option<SelectedModelDirectory>, String> {
    let config = load_config(&app)?;
    if !matches!(config.mode, RuntimeMode::Local) {
        return Err("the native directory picker is available only for the local server".into());
    }
    let selected = rfd::AsyncFileDialog::new()
        .set_title("Select a folder containing MFQ models")
        .set_directory(initial_directory)
        .pick_folder()
        .await;
    let Some(directory) = selected else {
        return Ok(None);
    };
    let path = directory
        .path()
        .to_str()
        .ok_or_else(|| "selected model directory is not valid UTF-8".to_string())?;
    let client = server_client(&local_service_url(&config), Duration::from_secs(300))?;
    let status = status_for(&app, config.clone()).await?;
    if !status.reachable { return Err("Connect to and confirm the server before registering a model directory.".into()); }
    let server = status.server.ok_or("The server is unavailable.")?;
    let token = studio_credential_get()?;
    let request = client
        .post(format!(
            "{}/api/v1/models/directories/register",
            local_service_url(&config)
        ))
        .header("X-MFQ-Instance-ID", server.instance_id)
        .json(&serde_json::json!({"path": path}));
    let response = with_server_credential(request, &token)
        .send()
        .await
        .map_err(|error| format!("failed to register model directory: {error}"))?;
    if !response.status().is_success() {
        let status = response.status();
        let detail = response.text().await.unwrap_or_default();
        return Err(format!(
            "model directory registration failed ({status}): {detail}"
        ));
    }
    let registered = response
        .json::<RegisteredModelList>()
        .await
        .map_err(|error| format!("invalid model registration response: {error}"))?;
    Ok(Some(SelectedModelDirectory {
        path: path.to_string(),
        names: registered
            .data
            .into_iter()
            .map(|model| model.name)
            .collect(),
    }))
}

#[tauri::command]
fn studio_confirm(message: String) -> bool {
    matches!(
        rfd::MessageDialog::new()
            .set_title("MFQ Studio")
            .set_description(message)
            .set_buttons(rfd::MessageButtons::YesNo)
            .show(),
        rfd::MessageDialogResult::Yes
    )
}

#[tauri::command]
fn studio_credential_get() -> Result<String, String> {
    let entry = keyring::Entry::new(CREDENTIAL_SERVICE, CREDENTIAL_ACCOUNT)
        .map_err(|error| error.to_string())?;
    match entry.get_password() {
        Ok(value) => Ok(value),
        Err(keyring::Error::NoEntry) => Ok(String::new()),
        Err(error) => Err(error.to_string()),
    }
}

#[tauri::command]
fn studio_credential_set(token: String) -> Result<(), String> {
    let entry = keyring::Entry::new(CREDENTIAL_SERVICE, CREDENTIAL_ACCOUNT)
        .map_err(|error| error.to_string())?;
    let token = token.trim();
    if token.is_empty() {
        match entry.delete_credential() {
            Ok(()) | Err(keyring::Error::NoEntry) => Ok(()),
            Err(error) => Err(error.to_string()),
        }
    } else {
        entry.set_password(token).map_err(|error| error.to_string())
    }
}

#[tauri::command]
fn studio_open_external(url: String) -> Result<(), String> {
    let parsed = Url::parse(&url).map_err(|error| format!("invalid external URL: {error}"))?;
    let host = parsed.host_str().unwrap_or_default().to_ascii_lowercase();
    if parsed.scheme() != "https"
        || !parsed.username().is_empty()
        || parsed.password().is_some()
        || parsed.port_or_known_default() != Some(443)
        || !matches!(
            host.as_str(),
            "github.com"
                | "www.github.com"
                | "huggingface.co"
                | "www.huggingface.co"
                | "modelscope.cn"
                | "www.modelscope.cn"
                | "www.deepseek.com"
                | "opencode.ai"
                | "code.claude.com"
                | "openclaw.ai"
        )
    {
        return Err("external links are limited to trusted MFQ model and release hosts".into());
    }
    #[cfg(target_os = "macos")]
    let mut command = Command::new("/usr/bin/open");
    #[cfg(target_os = "windows")]
    let mut command = {
        let mut value = Command::new("rundll32.exe");
        value.arg("url.dll,FileProtocolHandler");
        value
    };
    #[cfg(all(unix, not(target_os = "macos")))]
    let mut command = Command::new("xdg-open");
    command
        .arg(url)
        .spawn()
        .map(|_| ())
        .map_err(|error| format!("could not open external link: {error}"))
}

fn main() {
    if updater::maybe_run_update_helper() {
        return;
    }
    tauri::Builder::default()
        .manage(StudioState::default())
        .manage(updater::UpdateState::default())
        .setup(|app| {
            #[cfg(target_os = "windows")]
            if let Some(window) = app.get_webview_window("main") {
                windows_icon::set(&window)?;
                window.set_decorations(false)?;
            }
            #[cfg(not(target_os = "windows"))]
            let _ = app;
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            studio_custom_titlebar,
            studio_status,
            studio_configure,
            studio_start_local,
            studio_connect_existing,
            studio_select_model_directory,
            studio_confirm,
            studio_credential_get,
            studio_credential_set,
            studio_open_external,
            updater::studio_update_status,
            updater::studio_update_progress,
            updater::studio_update_set_automatic,
            updater::studio_update_set_download,
            updater::studio_update_download,
            updater::studio_update_install,
            updater::studio_update_delete
        ])
        .run(tauri::generate_context!())
        .expect("MFQ Studio failed");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    fn pipe_child() -> Child {
        Command::new("/bin/cat").stdin(Stdio::piped()).stdout(Stdio::null()).spawn().unwrap()
    }

    #[cfg(windows)]
    fn pipe_child() -> Child {
        let root = std::env::var_os("SystemRoot").expect("Windows system directory");
        let mut command = Command::new(PathBuf::from(root).join("System32/sort.exe"));
        command.stdin(Stdio::piped()).stdout(Stdio::null());
        configure_detached(&mut command);
        command.spawn().unwrap()
    }

    fn wait_for_pipe_child(child: &mut Child) {
        for _ in 0..100 {
            if let Some(status) = child.try_wait().unwrap() {
                assert!(status.success());
                return;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        let _ = child.kill();
        let _ = child.wait();
        panic!("owned server did not receive lifecycle EOF");
    }

    #[test]
    fn dropping_studio_closes_only_its_owned_server_pipe() {
        let mut owned = pipe_child();
        let mut external = pipe_child();
        let state = StudioState::default();
        state.own_server(&mut owned, "owned-instance").unwrap();
        assert!(owned.stdin.is_none());
        assert!(owned.try_wait().unwrap().is_none());
        drop(state);
        wait_for_pipe_child(&mut owned);
        assert!(external.try_wait().unwrap().is_none());
        drop(external.stdin.take());
        wait_for_pipe_child(&mut external);
    }

    #[test]
    fn startup_cleanup_does_not_close_a_different_owned_server() {
        let mut owned = pipe_child();
        let state = StudioState::default();
        state.own_server(&mut owned, "owned-instance").unwrap();
        state.release_server(owned.id().wrapping_add(1));
        assert!(owned.try_wait().unwrap().is_none());
        state.release_server(owned.id());
        wait_for_pipe_child(&mut owned);
    }

    #[test]
    fn update_shutdown_waits_for_owned_server_and_preserves_external_server() {
        let state = StudioState::default();
        let mut owned = pipe_child();
        let mut external = pipe_child();
        state.own_server(&mut owned, "owned-instance").unwrap();
        *state.server_child.lock().unwrap() = Some(owned);
        state.stop_owned_server().unwrap();
        assert!(state.server_child.lock().unwrap().is_none());
        assert!(state.server_input.lock().unwrap().is_none());
        state.stop_owned_server().unwrap();
        assert!(external.try_wait().unwrap().is_none());
        drop(external.stdin.take());
        wait_for_pipe_child(&mut external);
    }

    fn startup_test_dir() -> PathBuf {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let nonce = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let sequence = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("mfq-startup-test-{}-{nonce}-{sequence}", std::process::id()));
        fs::create_dir(&path).unwrap();
        path
    }

    fn test_identity() -> ServerIdentity {
        ServerIdentity { service: "mfq-server".into(), protocol_version: "1.0".into(),
            studio_protocol_version: "1.0".into(), version: env!("MFQ_SERVER_VERSION").into(),
            instance_id: uuid::Uuid::new_v4().to_string(), workspace_id: "a".repeat(64), pid: 123,
            launch_mode: "cli".into() }
    }

    #[test]
    fn identity_requires_compatible_versions_and_real_instance_fields() {
        let valid = test_identity();
        valid.validate().unwrap();
        for invalid in [ServerIdentity { version: "0.0.0".into(), ..valid.clone() },
            ServerIdentity { protocol_version: "2.0".into(), ..valid.clone() },
            ServerIdentity { studio_protocol_version: String::new(), ..valid.clone() },
            ServerIdentity { instance_id: String::new(), ..valid.clone() },
            ServerIdentity { workspace_id: "invalid".into(), ..valid.clone() },
            ServerIdentity { pid: 0, ..valid.clone() },
            ServerIdentity { launch_mode: "unknown".into(), ..valid.clone() },
            ServerIdentity::default()] {
            assert!(invalid.validate().is_err(), "{invalid:?}");
        }
    }

    #[test]
    fn external_confirmation_is_bound_to_endpoint_instance_and_workspace() {
        let state = StudioState::default();
        let server = test_identity();
        let url = "http://127.0.0.1:8090";
        assert!(!state.accepts(url, &server));
        *state.external_server.lock().unwrap() = Some((url.into(), server.clone()));
        assert!(state.accepts(url, &server));
        assert!(!state.accepts("http://127.0.0.1:8092", &server));
        assert!(!state.accepts(url, &ServerIdentity { instance_id: uuid::Uuid::new_v4().to_string(), ..server.clone() }));
        assert!(!state.accepts(url, &ServerIdentity { workspace_id: "b".repeat(64), ..server.clone() }));
        assert!(!state.owns(&server));
    }

    #[test]
    fn ownership_needs_both_spawned_pid_and_instance_id_and_clears_after_exit() {
        let state = StudioState::default();
        let mut child = pipe_child();
        let server = ServerIdentity { pid: child.id(), launch_mode: "studio".into(), ..test_identity() };
        state.own_server(&mut child, &server.instance_id).unwrap();
        assert!(state.owns(&server));
        assert!(!state.owns(&ServerIdentity { instance_id: uuid::Uuid::new_v4().to_string(), ..server.clone() }));
        assert!(!state.owns(&ServerIdentity { launch_mode: "cli".into(), ..server.clone() }));
        child.kill().unwrap();
        child.wait().unwrap();
        *state.server_child.lock().unwrap() = Some(child);
        state.reap_exited_server();
        assert!(!state.owns(&server));
    }

    #[test]
    fn only_loopback_server_connections_bypass_proxies() {
        for url in ["http://localhost:8090", "https://LOCALHOST:8090", "http://127.0.0.1:8090", "http://127.0.0.2:8090", "http://[::1]:8090"] {
            assert!(is_loopback_service(url), "{url}");
        }
        for url in ["https://example.com", "http://localhost.example.com", "http://192.168.1.2:8090", "http://[::2]:8090", "not a URL"] {
            assert!(!is_loopback_service(url), "{url}");
        }
    }

    #[test]
    fn startup_failure_reports_only_this_launch_and_bounds_log_output() {
        let directory = startup_test_dir();
        let path = directory.join(LOG_FILE);
        let previous = "previous-launch-private-text\n";
        fs::write(&path, format!("{previous}new launch: library not found\n")).unwrap();
        let error = startup_failure(&path, previous.len() as u64, "Server exited.");
        assert!(error.contains("Server exited."));
        assert!(error.contains("library not found"));
        assert!(!error.contains("previous-launch-private-text"));
        fs::write(&path, format!("{}\nlast error", "x".repeat(20000))).unwrap();
        let error = startup_failure(&path, 0, "Server exited.");
        assert!(error.contains("last error"));
        assert!(error.len() < 8500);
        assert!(startup_failure(&path, 30000, "Server exited.").contains("Server log:"));
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn startup_failure_keeps_the_summary_when_the_log_is_missing() {
        let directory = startup_test_dir();
        let error = startup_failure(&directory.join(LOG_FILE), 0, "Server exited.");
        assert!(error.starts_with("Server exited.\nServer log:"));
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn startup_failure_after_log_rotation_does_not_expose_the_previous_log() {
        use std::io::Write;
        let directory = startup_test_dir();
        let path = directory.join(LOG_FILE);
        fs::write(&path, "previous-launch-private-text").unwrap();
        OpenOptions::new().write(true).open(&path).unwrap().set_len(LOG_ROTATE_BYTES).unwrap();
        let mut log = open_log(&path).unwrap();
        let offset = log.metadata().unwrap().len();
        assert_eq!(offset, 0);
        writeln!(log, "current launch error").unwrap();
        drop(log);
        let error = startup_failure(&path, offset, "Server exited.");
        assert!(error.contains("current launch error"));
        assert!(!error.contains("previous-launch-private-text"));
        assert_eq!(path.with_extension("log.1").metadata().unwrap().len(), LOG_ROTATE_BYTES);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn startup_rejects_shared_api_ports_without_spawning_a_server() {
        let directory = startup_test_dir();
        fs::write(directory.join("listener.json"), r#"{"anthropic_port":19290}"#).unwrap();
        let error = check_local_ports(&directory, 19290).unwrap_err();
        assert!(error.contains("cannot share port 19290"));
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn startup_identifies_the_occupied_api_port() {
        let directory = startup_test_dir();
        let occupied = std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap();
        let occupied_port = occupied.local_addr().unwrap().port();
        let available = std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap();
        let available_port = available.local_addr().unwrap().port();
        drop(available);
        fs::write(directory.join("listener.json"), format!(r#"{{"anthropic_port":{occupied_port}}}"#)).unwrap();
        let error = check_local_ports(&directory, available_port).unwrap_err();
        assert!(error.contains(&format!("Anthropic API port {occupied_port} is already in use")));
        fs::write(directory.join("listener.json"), format!(r#"{{"anthropic_port":{available_port}}}"#)).unwrap();
        let error = check_local_ports(&directory, occupied_port).unwrap_err();
        assert!(error.contains(&format!("OpenAI API port {occupied_port} is already in use")));
        fs::remove_dir_all(directory).unwrap();
    }

    #[cfg(any(target_os = "macos", target_os = "windows"))]
    #[test]
    fn desktop_credentials_use_a_native_store_instead_of_the_mock_backend() {
        let builder = keyring::default::default_credential_builder();
        assert!(!builder.as_any().is::<keyring::mock::MockCredentialBuilder>());
    }

    #[test]
    fn server_requests_use_the_trimmed_credential_without_empty_auth_headers() {
        let client = reqwest::Client::new();
        for path in ["/api/v1/runtime/listener", "/api/v1/models/directories/register"] {
            let url = format!("http://127.0.0.1:8090{path}");
            let authenticated = with_server_credential(client.post(&url), " test-credential ").build().unwrap();
            assert_eq!(authenticated.headers()[reqwest::header::AUTHORIZATION], "Bearer test-credential");
            assert!(authenticated.headers()[reqwest::header::AUTHORIZATION].is_sensitive());
            for empty in ["", " \n "] {
                let anonymous = with_server_credential(client.post(&url), empty).build().unwrap();
                assert!(!anonymous.headers().contains_key(reqwest::header::AUTHORIZATION));
            }
        }
    }
}
