#!/usr/bin/env bash

mfq_add_release_path() {
  local mfq_source="$1" mfq_destination="$2" mfq_spelling mfq_flag
  [[ -n "${mfq_source}" ]] || return 0
  [[ "${mfq_source}" == /* ]] || mfq_source="${mfq_studio_dir}/${mfq_source}"
  local mfq_resolved
  mfq_resolved="$("${mfq_venv_dir}/bin/python" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "${mfq_source}")"
  [[ "${mfq_resolved}" != / ]] || { printf 'error: refusing to remap the filesystem root\n' >&2; return 1; }
  for mfq_spelling in "${mfq_source%/}" "${mfq_resolved}"; do
    mfq_private_prefix_args+=(--private-prefix "${mfq_spelling}")
    CARGO_ENCODED_RUSTFLAGS+="${mfq_rust_separator}--remap-path-prefix=${mfq_spelling}=${mfq_destination}"
    mfq_flag="$("${mfq_venv_dir}/bin/python" -c 'import shlex, sys; print(shlex.quote(sys.argv[1]))' "-ffile-prefix-map=${mfq_spelling}=${mfq_destination}")"
    mfq_release_native_flags+=" ${mfq_flag}"
    [[ "${mfq_spelling}" != "${mfq_resolved}" ]] || break
  done
}

mfq_configure_release_paths() {
  mfq_rust_separator=$'\x1f'
  if [[ -z "${CARGO_ENCODED_RUSTFLAGS+x}" ]]; then
    CARGO_ENCODED_RUSTFLAGS="$("${mfq_venv_dir}/bin/python" -c 'import os; print("\x1f".join(os.environ.get("RUSTFLAGS", "").split()), end="")')"
  fi
  mfq_private_prefix_args=()
  mfq_release_native_flags=""
  mfq_add_release_path "${HOME}" /mfq-build/home
  mfq_add_release_path "$("${mfq_venv_dir}/bin/python" -c 'import sys; print(sys.base_prefix)')" /mfq-build/python
  mfq_add_release_path "${mfq_venv_dir}" /mfq-build/venv
  mfq_add_release_path "${CARGO_HOME:-${HOME}/.cargo}" /mfq-build/cargo
  mfq_add_release_path "${RUSTUP_HOME:-${HOME}/.rustup}" /mfq-build/rustup
  mfq_add_release_path "${CARGO_TARGET_DIR:-${mfq_studio_dir}/src-tauri/target}" /mfq-build/target
  mfq_add_release_path "${mfq_project_dir}" /mfq-src
  CARGO_ENCODED_RUSTFLAGS="${CARGO_ENCODED_RUSTFLAGS#"${mfq_rust_separator}"}"
  OBJCXXFLAGS="${OBJCXXFLAGS:-${CXXFLAGS:-}}${mfq_release_native_flags}"
  CFLAGS="${CFLAGS:-}${mfq_release_native_flags}"
  CXXFLAGS="${CXXFLAGS:-}${mfq_release_native_flags}"
  export CARGO_ENCODED_RUSTFLAGS CFLAGS CXXFLAGS OBJCXXFLAGS
  export CC_SHELL_ESCAPED_FLAGS=1
}
