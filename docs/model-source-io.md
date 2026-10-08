# Model source file I/O

`MFQ_MODEL_FILE_IO=buffered` is the default. `MFQ_MODEL_FILE_IO=direct`
selects owning, positional file readers for MFQ tensor payloads and assets,
including retained tensor callbacks. Unsupported modes or direct-I/O failures
are reported; they do not silently fall back to buffered payload reads.

On Windows, each shard has a shared `NO_BUFFERING | OVERLAPPED` handle.
Sector alignment is queried from the file. Unaligned ranges and destinations
use temporary aligned storage that is released after the request; requests
have independent offsets and completion events. The reader retains no payload
cache. It validates exact logical bounds, including the final partial sector,
and callbacks remain usable after the originating model source is destroyed.
Bounded header parsing continues to use buffered reads. The POSIX `O_DIRECT`
and macOS `F_NOCACHE` implementations require validation on those platforms.

`MFQ_REPORT_FILE_IO=1` enables optional native generation diagnostics at
prefill and decode boundaries. They report logical calls/bytes, bytes returned
by the file API, errors, active temporary staging, summed per-shard staging
peaks, and cumulative reader time. With concurrent reads, cumulative reader
time can exceed elapsed time. Returned bytes are not a device-level SSD traffic
counter, and summed per-shard peaks are not a simultaneous global memory peak.

Direct I/O avoids the OS payload file cache but does not make process commit
equal to physical host RAM. GPU driver backing, pinned transfers, runtime
metadata, activations, managed expert caches and transient storage must also
be accounted for when reporting a memory-constrained benchmark. Whole-model
comparisons must state the I/O mode and use identical weights, token IDs,
generation length, GPU cache budget and profiling settings.
