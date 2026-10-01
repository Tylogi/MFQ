"""校验 Tauri Rust、发布脚本与 TypeScript 桥接的跨语言契约。

纯前端迁移映射：MFQStudio/tests/studioContracts.test.ts、
studioBehavior.test.tsx、studioMedia.test.tsx 与 voiceContracts.test.ts。
Vitest 中保留旧测试名；源码契约明确标注为过渡，媒体等行为单独验证。
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDIO = ROOT / "MFQStudio"
TAURI = STUDIO / "src-tauri"
RUST = (TAURI / "src" / "main.rs").read_text(encoding="utf-8")
BUILD = (TAURI / "build.rs").read_text(encoding="utf-8")
STUDIO_BRIDGE = (STUDIO / "src" / "studio.ts").read_text(encoding="utf-8")
PLATFORM_BRIDGE = STUDIO / "src" / "shared" / "platform" / "studio.ts"
if PLATFORM_BRIDGE.exists():
    STUDIO_BRIDGE += "\n" + PLATFORM_BRIDGE.read_text(encoding="utf-8")
MODEL_BROWSER = (
    STUDIO / "src" / "features" / "models" / "ModelBrowser.tsx"
).read_text(encoding="utf-8")
UPDATE_MANAGER = (
    STUDIO / "src" / "features" / "settings" / "UpdateManager.tsx"
).read_text(encoding="utf-8")
UPDATER = (TAURI / "src" / "updater.rs").read_text(encoding="utf-8")
RELEASE_SCRIPT = (ROOT / "packaging" / "build_release_mac.sh").read_text(encoding="utf-8")


def test_studio_uses_one_package_for_web_and_desktop_clients():
    config = json.loads((TAURI / "tauri.conf.json").read_text(encoding="utf-8"))
    release_config = json.loads(
        (TAURI / "tauri.release-macos.conf.json").read_text(encoding="utf-8")
    )
    package = json.loads((STUDIO / "package.json").read_text(encoding="utf-8"))
    assert not (STUDIO / "web").exists()
    assert not (STUDIO / "desktop").exists()
    assert package["name"] == "@mfq/studio"
    assert package["scripts"]["tauri"] == "tauri"
    assert config["build"]["frontendDist"] == "../dist"
    assert config["build"]["beforeBuildCommand"] == "npm run build"
    assert config["identifier"] == "com.tylogi.mfq-studio"
    assert "icons/icon.ico" in config["bundle"]["icon"]
    assert "icons/icon.icns" in config["bundle"]["icon"]
    assert "IconDir::new" in BUILD
    assert "IconFamily::new" in BUILD
    assert "media-src 'self' asset: data: blob:" in config["app"]["security"]["csp"]
    assert release_config["bundle"]["macOS"]["hardenedRuntime"] is False
    assert 'if [[ "${mfq_signing_identity}" != "-" ]]' in RELEASE_SCRIPT
    assert '"hardenedRuntime":true' in RELEASE_SCRIPT
    assert "--remap-path-prefix=${HOME}=/mfq-build/home" in RELEASE_SCRIPT
    assert "--remap-path-prefix=${mfq_project_dir}=/mfq-src" in RELEASE_SCRIPT
    assert 'RUSTFLAGS="${mfq_release_rustflags}"' in RELEASE_SCRIPT
    assert "packaged Studio contains a private build path" in RELEASE_SCRIPT
    assert 'mfq-decode-metal" --self-test-metal' in RELEASE_SCRIPT


def test_studio_starts_the_unified_local_server_and_bundled_runtime():
    assert 'Some("mfq-server")' in RUST
    assert "Command::new" in RUST
    assert "studio_start_local" in RUST
    assert '.arg("serve")' in RUST
    assert '.arg("--data-dir")' in RUST
    assert '.arg("--db")' not in RUST
    assert 'command.arg("--running-executable")' in RUST
    assert "MFQ_MLX_METALLIB" in RUST
    assert "MFQ_AVFOUNDATION_VIDEO_LIBRARY" in RUST


def test_studio_supports_local_and_remote_server_connections_with_voice_controls():
    for command in ('studio_configure', 'studio_start_local', 'studio_select_model_directory'):
        assert command in RUST


def test_studio_can_select_and_load_an_external_mfq_directory_in_local_mode():
    assert 'rfd::AsyncFileDialog::new()' in RUST
    assert '.pick_folder()' in RUST
    assert "tauri.invoke<string[] | null>('studio_select_model_directory')" in STUDIO_BRIDGE


def test_studio_uses_native_confirmation_dialogs_for_destructive_actions():
    assert "fn studio_confirm(message: String) -> bool" in RUST
    assert "rfd::MessageButtons::YesNo" in RUST
    assert "studio_confirm," in RUST
    assert "tauri.invoke<boolean>('studio_confirm', { message })" in STUDIO_BRIDGE


def test_model_hub_resolves_links_and_downloads_selected_variants():
    assert "modelsApi.resolveHubModel(reference, provider)" in MODEL_BROWSER
    assert (
        "destination: `models/${source.provider}/${repositoryPath}/${variantPath}`"
        in MODEL_BROWSER
    )
    assert "downloadPatterns(variant)" in MODEL_BROWSER
    assert "jobsApi.createJob" in MODEL_BROWSER
    assert 'tr("官方模型", "Official")' in MODEL_BROWSER
    assert 'tr("第三方模型", "Community")' in MODEL_BROWSER
    assert 'tr("内存压力", "Memory pressure")' in MODEL_BROWSER
    assert "not model capability or quality" in MODEL_BROWSER
    assert "三星推荐" not in MODEL_BROWSER
    assert "3-star recommendation" not in MODEL_BROWSER
    for symbol in ('"★★★"', '"★★"', '"★"', '"▲"', '"✕"'):
        assert symbol in MODEL_BROWSER


def test_studio_checks_releases_and_keeps_verified_versions_for_rollback():
    assert "studio_update_status" in RUST
    assert "studio_update_download" in RUST
    assert "studio_update_install" in RUST
    assert "studio_update_delete" in RUST
    assert "studio_update_progress" in RUST
    assert "api.github.com/repos/Tylogi/TyloQuant/releases" in UPDATER
    assert "release SHA-256 verification failed" in UPDATER
    assert "snapshot_current_bundle" in UPDATER
    assert "replace_macos_bundle" in UPDATER
    assert "com.tylogi.mfq-studio" in UPDATER
    assert "studioUpdateStatus(false)" in UPDATE_MANAGER
    assert 'tr("自动检查并提醒", "Automatically check and notify")' in UPDATE_MANAGER
