# NVQ fixed-table search and packing

The CUDA NVQ-JSC path keeps the existing quantization objective, codebooks,
anchor refinements, tie-breaking and serialized format. Both unweighted and
imatrix-weighted quantization use the optimized path; column and per-weight
importance arrays are still passed to every search/refit.

## Changes

- Fixed-table generic assignment reuses the floating group scales from its
  initialization. These depend only on weights, importance and the fixed bank;
  changing a row anchor does not change this search. With two banks and two
  anchor refinements (NVQ3J-512/L), this reduces eight searches to two. All six
  fixed-scale reassignments are retained. No cache is shared between calls or
  across codebook-training updates.
- Row-aligned CUDA search batches are submitted from C++, including fused
  NVQ2J-L/XL training search. NVQ2J/3J bank validation reads the device table
  once per invocation instead of once per batch. The existing search kernels
  and arithmetic are unchanged.
- Explicit CUDA packing writes the existing bitstreams and NVQ2J-XL group64
  records without expanding the data to individual CPU bits. HF/GGUF CUDA
  streaming exporters use it automatically. No Triton dependency is added.

`group_chunk` is now honored by the CUDA JSC paths. The submission size is
rounded down to whole rows, with at least one row, and capped at 4096 actual
groups. A row containing more than 4096 groups is rejected before launching.
The cap addresses observed device failures with large search submissions; it
does not change the format's 24-weight quantization group size. Smaller targets
than one row are rounded up to one row to preserve anchor-refit reductions.

CPU/Metal export retains CPU packing. Library callers may explicitly select
CUDA packing with `pack_nvq(tensor, device="cuda:0")`; the default remains CPU
and does not import Torch or initialize CUDA. Shape/range/metadata validation
remains in the canonical format serializer.

## Numerical regression tests

```sh
python -m pytest -q \
  tests/test_quantize/test_nvq_search_packing.py \
  tests/test_quantize/test_nvq_jsc.py \
  tests/test_formats/test_nvq.py tests/test_formats/test_nvq_compat.py \
  tests/test_quantize/test_hf_to_mfq.py tests/test_quantize/test_gguf_to_mfq.py
```

Tests cover all six NVQ2J/3J profiles at input widths 640 and 2560; unweighted,
column-imatrix and per-weight-imatrix objectives; zero rows/importance; tail
padding; 2048/4096-group submission targets; cached versus recomputed scales;
fused training search; invalid bank mappings; all 1–16-bit packing widths;
partial group64 vectors; and imatrix-weighted CUDA export versus CPU packing.
Anchors, states, indices, signs and serialized bytes must be exactly equal.

These are kernel/serializer regression tests, not a full-model quality or
throughput claim. Model acceptance still requires full WT2 forward KLD and
Top1 agreement under the same evaluation contract.
