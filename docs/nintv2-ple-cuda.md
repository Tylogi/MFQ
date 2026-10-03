# CUDA NINTv2 PLE selected-row decoding

Native CUDA Flash-Next PLE now uses a retained, bounded MFQ file-range reader
for NINT shards instead of loading the complete shard into QuantLinear.
The shared C++ `NintRows` parser owns only q/k selectors and compact rank
checkpoints in this mode (below one byte per embedding row). Packed q,
subgroup scale/min and FP16 row anchors remain in the file. The reader owns
its file handle independently of the source object and serializes seek/read
operations; it does not mmap or register the full table with CUDA.

Each lookup transfers requested token IDs to the host, gathers only selected
packed row byte ranges, uploads the common six-word row descriptors and runs
`nint_selected_rows_kernel`. Every row has independent q/k and group size;
the kernel reads both compressed streams directly and returns F16. The old
resident NINT embedding operator remains available for other consumers.
The canonical wire format, relative-k selector range, quantizer policy and
existing non-NINT PLE paths are unchanged. NINT conversion is still opt-in
with `--quantize-ple`; existing trained-model artifacts are not converted.

The range-backed lookup needs a device-to-host ID transfer and CPU file I/O,
so it explicitly rejects CUDA graph capture. The pure decoder supports
capture with already staged rows, but this does not establish graph-captured
demand lookup. Repeated IDs currently repeat reads; no row cache/deduplication,
cold-SSD latency, whole-model speedup or quantization-quality gain is claimed.

## NVIDIA validation

From the repository root on Linux with CUDA installed:

```sh
cmake -S cpp_runtime -B build/cuda -DCMAKE_BUILD_TYPE=Release -DMFQ_BUILD_CPP_SERVER=ON -DMFQ_BUILD_METAL_RUNTIME=OFF -DMFQ_CUDA_ARCHITECTURES=native -DBUILD_TESTING=ON
cmake --build build/cuda --target mfq-diagnostics mfq-nint-rows-cuda-test mfq-nint-rows-test -j 8
ctest --test-dir build/cuda --output-on-failure -V -R '^mfq-nint-rows(-cuda)?-test$'
MFQ_NINT_PLE_CUDA_DIAGNOSTICS="$PWD/build/cuda/mfq-diagnostics" python -m pytest -q tests/test_native_nint_ple.py
```

For multi-config Windows generators, add `--config Release` to the build,
`-C Release` to CTest, and set `MFQ_NINT_PLE_CUDA_DIAGNOSTICS` to the actual
`Release/mfq-diagnostics.exe` path using PowerShell's `$env:` syntax.

The CUDA target checks all q1--q8/k1--k8 across legal nominal-k families,
uniform packed storage, unaligned/tail rows, checkpoint boundaries, duplicate
IDs, different group sizes across tables, fractional/negative/subnormal
anchors, an empty batch and staged-decoder graph replays where supported.
Exit 77 is a skipped test (no usable GPU), not a pass.
The Python gate writes actual canonical MFQ fixtures and compares dense-F16
versus NINTv2 native Flash-Next logits for prefill, chunked decode, EOS, batch,
reset and speculative commit/rollback. Both models use identical exactly
representable PLE values, so every output must match exactly.

Local macOS validation covers the shared range parser, retained source
lifetime/bounds, production geometry demand access, and Metal regression.
CUDA compilation/execution must still be verified on the NVIDIA host.
