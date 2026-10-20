# `mfq build`

`mfq build` compiles the native inference runtime for the current machine. It
detects Metal or CUDA, invokes CMake, and records the resulting executable so
that [`mfq serve`](serve.md) can find it without a binary path.

The `mfq` command must come from an MFQ source checkout containing
`csrc/CMakeLists.txt`. MFQ derives the repository root from the installed
Python package, not from the current working directory. In an editable
development environment, run the CLI through uv from that checkout.

## Quick start

Sync the dependency set for the current host once:

```shell
cd /path/to/MFQ
uv sync --extra metal
```

On a CUDA host, the native build itself needs no Python accelerator package:

```shell
uv sync
```

Build with automatic backend detection:

```shell
uv run mfq build
```

Automatic backend selection uses:

- Metal on Apple silicon macOS;
- CUDA on Linux or Windows when `nvcc` is available.

Backend detection does not check every compiler dependency. Requirements are
listed below. Pass the backend explicitly when debugging a build:

```shell
uv run mfq build --backend metal
uv run mfq build --backend cuda
```

## Output and build manifest

`mfq build` compiles the inference runtime only. The default CMake build
directory is `<repo>/build/csrc`, where `<repo>` is this `cuda` checkout; from
the MFQ workspace root that directory is `cuda/build/csrc`. Executable paths:

| Backend | Executable |
| --- | --- |
| Metal | `<repo>/build/csrc/metal/mfq-decode-metal` |
| CUDA | `<repo>/build/csrc/mfq-runtime` |
| CUDA on Windows | Usually `<repo>\build\csrc\mfq-runtime.exe`; multi-configuration generators may add `Release\` |

`mfq serve` needs nothing else; normal serving works with the runtime alone.
MFQ searches the build tree after compilation, so generators may place the
executable in a configuration subdirectory.

A successful build writes `<repo>/build/mfq-runtime.json`. This location is
fixed relative to the source checkout even when `--build-dir` points elsewhere.
The manifest stores the backend, source directory, build directory, build type,
generator, forwarded CMake arguments, target, and the actual absolute
executable path. It is internal CLI state under the ignored `build/` directory;
users do not add the native executable to `PATH`.

To inspect a generator-dependent path, read the `executable` field for the
selected backend in the manifest:

```shell
uv run -m json.tool build/mfq-runtime.json
```

`mfq serve` reads this manifest automatically. If the executable exists, it is
used directly. If it was deleted, MFQ rebuilds it in the recorded directory with
the recorded configuration.

### Optional native tools

The CUDA check, benchmark, and evaluation tools are separate CMake targets and
are not built by default. `mfq build` has no target option, so build a tool
with CMake in the build directory you use:

```shell
cmake --build build/csrc --config Release --target mfq-diagnostics
cmake --build build/csrc --config Release --target mfq-eval

build/csrc/mfq-diagnostics --check-deepseek-v41
build/csrc/mfq-eval kl --model model.mfq --kl-base reference.bin
```

Use the directory and configuration of your own build. `--config` matters for
multi-configuration generators such as Visual Studio, which place the
executable in `<build-dir>\Release\`.

CUDA KL evaluation is available only through the `mfq-eval kl` command line: a
CUDA server does not offer the Metal-only tensor kernel benchmark or WikiText-2
quality interfaces. A CUDA `model.validate` job reports `tool_unavailable` with
the `mfq-diagnostics` CMake command when that executable is missing; the server
keeps serving and other jobs keep running.

## Building without a device backend

`mfq build` always builds one inference backend. CMake alone can configure the
common engine with both backends off: that configuration does not look for the
CUDA Toolkit, and the CPU targets it builds invoke no `nvcc` and link no CUDA
or MLX library:

```shell
cmake -S csrc -B build/common -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DMFQ_BUILD_CUDA_RUNTIME=OFF -DMFQ_BUILD_METAL_RUNTIME=OFF
cmake --build build/common --target mfq-engine-contract-test
build/common/tests/mfq-engine-contract-test
```

`MFQ_BUILD_CUDA_RUNTIME` defaults to on when a CUDA compiler is detected on a
non-Apple platform and HIP is not selected; otherwise it defaults to off. It is the
only switch that enables the CUDA language, finds the CUDA Toolkit, or adds the
device tests. `mfq build --backend cuda` passes `-DMFQ_BUILD_CUDA_RUNTIME=ON`
explicitly, so a build directory previously configured with the switch off still
produces `mfq-runtime`.

## Custom build directory

Use `--build-dir` when the build tree belongs on another disk:

```shell
cd /path/to/MFQ
uv run mfq build --build-dir /mnt/fast-build/mfq
```

The manifest records the absolute build directory and executable path. A later
`mfq serve` command finds this custom build without another option:

```shell
uv run mfq serve --model /models/model.mfq
```

Both commands must use the `mfq` installation associated with the same source
checkout. The current working directory may change between them.

## Forward CMake configuration arguments

Arguments after `--` are passed to the CMake configure command. The separator
is required so the arguments are not parsed as `mfq build` options.

```shell
uv run mfq build --backend cuda -- \
  -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DGGML_CCACHE=OFF
```

The build manifest saves these arguments and reuses them when rebuilding a
missing executable.

## Command options

| Option | Meaning | Default |
| --- | --- | --- |
| `--backend {auto,cuda,metal}` | Select or detect the accelerator backend. | `auto` |
| `--build-dir PATH` | Set the CMake build directory. | `<repo>/build/csrc` |
| `--build-type TYPE` | Set `CMAKE_BUILD_TYPE` and the build configuration. | `Release` |
| `-j N`, `--jobs N` | Set parallel build jobs. | Host CPU count |
| `--generator NAME` | Select a CMake generator. | Ninja when available, otherwise CMake default |
| `--dry-run` | Print configure and build commands without executing them. | Off |
| `-- CMAKE_ARG...` | Forward remaining arguments to CMake configuration. | None |

`--dry-run` does not compile an executable or update the manifest.

## Backend requirements

### Metal

- Apple silicon macOS;
- CMake 3.26 or newer;
- a C++ toolchain;
- the `metal` optional dependency, which provides MLX and its native assets.

For a source checkout managed by uv:

```shell
uv sync --extra metal
uv run mfq build
```

### CUDA

- Linux or Windows;
- CMake 3.26 or newer;
- a C++/CUDA toolchain with `nvcc`;
- CUDA Toolkit 12 or newer, including `nvcc`, cuBLAS, and the CUDA runtime.

The default CUDA runtime needs no Python, PyTorch, or LibTorch. Migration A/B
builds may set `-DMFQ_BUILD_TORCH_REFERENCE_RUNTIME=ON`; that reference target
has separate Python and LibTorch requirements.

MFQ checks `CUDACXX`, `PATH`, `CUDA_HOME`, and `CUDA_PATH` when locating
`nvcc`.

## Troubleshooting

### No supported accelerator was detected

Confirm the host is Apple silicon, or make `nvcc` visible through one of the
CUDA locations above. `--backend` validates the requested backend; it does not
bypass platform and toolchain checks.

### CMake completed without producing the executable

Check the printed configure and build commands. MFQ searches custom generator
layouts by executable name; if the search still fails, the target was not
produced.

### Inspect the exact commands

```shell
uv run mfq build --dry-run --backend cuda -- \
  -DCMAKE_CUDA_ARCHITECTURES=90
```

Next: [`mfq serve`](serve.md).
