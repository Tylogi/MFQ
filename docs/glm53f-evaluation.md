# GLM-5.3-Flash WT2 evaluation

Use the original HF checkpoint to generate a reference, then evaluate an MFQ
file against it. The reference uses full 512-token windows, fresh KDA/DSA
state per window, and all four mHC streams. It runs the configured text
layers only (45 for the published model). Vision and the MTP layer do not
participate. The FP8 import preserves the stored E4M3 codes and 128×128 scales;
absorbed MLA head matrices are derived lazily from the original K/V projection.
Floating projections and the output head retain their checkpoint dtype.

On Windows, run these commands in a Visual Studio 2022 Developer PowerShell
with CUDA 13.0, CMake, Ninja, and the project's Python dependencies available.
From the repository root, configure and build both executables:

```powershell
cmake -S cpp_runtime -B build/eval -G Ninja -DCMAKE_BUILD_TYPE=Release "-DCMAKE_CUDA_COMPILER=$env:CUDA_PATH/bin/nvcc.exe" -DCMAKE_CUDA_ARCHITECTURES=120 -DMFQ_CUDA_ARCHITECTURES=120 -DMFQ_BUILD_RUNTIME_COMMUNICATION=ON -DMFQ_BUILD_METAL_RUNTIME=OFF -DMFQ_BUILD_TORCH_REFERENCE_RUNTIME=OFF
if ($LASTEXITCODE -ne 0) { throw 'CMake configuration failed' }
cmake --build build/eval --target mfq-runtime mfq-eval --parallel 4
```

Set paths to the original HF directory, the official `mfq-wt2-text-v1` corpus
already on disk, the desired reference output, and the quantized MFQ file:

```powershell
$HfModel = 'D:/models/GLM-5.3-Flash'
$Corpus = 'D:/datasets/wikitext-corpus.txt'
$Reference = 'D:/eval/glm53f-wt2.bin'
$MfqModel = 'D:/models/glm53f.mfq'
$N = 580
$env:PYTHONPATH = (Get-Location).Path
python -m mfq.cli wt2-reference --model $HfModel --dataset $Corpus --output $Reference --backend cuda --device cuda:0 --runtime ./build/eval/mfq-runtime.exe --context-size 512 --chunks $N
```

The corpus checksum is validated. Choose `N` within the tokenized corpus;
requesting too many windows fails. Layerwise generation is the default and
requires one layer's weights plus runtime workspace in VRAM. Do not request
`--resident` for the full 306 GiB source on a 96 GB GPU. The reference and
scratch require about 65.44 GB for 580 chunks; see the
[disk formulas](cli/utilities.md#mfq-wt2-reference) for other counts.

Evaluate all stored chunks with a bounded expert cache (GiB):

```powershell
./build/eval/mfq-eval.exe kl --model $MfqModel --kl-base $Reference --ctx-size 512 --kl-reference-n-batch 512 --kl-reference-n-ubatch 512 --moe-gpu-cache-gb 64
```

The shared expert cache supports packed NINT/NVQ/MFE experts and uses RAM or
file-backed expert ranges outside the GPU cache. Its 64 GiB budget leaves
room on a 96 GB GPU for non-expert weights, state, and workspaces; adjust it
to the actual artifact and available memory. Optional vision and MTP weights
in the MFQ file are ignored by KL evaluation. `cpp_kl_result` reports `kld`,
`same_top` (Top1 agreement), `bf16_ppl` (reference PPL, also for FP8 sources),
and `mfq_ppl`. Geometry flags enforce one full 512-token sequence per batch.

For native GPU regression tests, after building the two executables:

```powershell
$env:MFQ_CUDA_REFERENCE_RUNTIME = (Resolve-Path ./build/eval/mfq-runtime.exe).Path
python -m pytest tests/test_native_wt2_reference.py tests/test_glm_wt2_reference.py -q
```

The synthetic HF fixture includes original FP8 codes/scales, four-stream mHC,
KDA, DSA, routed/shared experts, and unused optional components. Tests compare
layerwise and resident execution, repeated-window state reset, the float32
handoff, and MFQ KL with mixed NINT/NVQ experts with and without expert offload.
GPU tests skip when no native executable or CUDA device is available. Windows
execution and full-checkpoint GPU metrics require verification on that host.
