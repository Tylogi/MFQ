#!/bin/sh
# Stage 0 capability matrix probe. One build at a time, serially, on one GPU.
# Usage: sh scripts/stage0-matrix.sh <build-dir> <out-dir>
# Every cell lands in $OUT/results.tsv with its command, exit code and status
# (pass | fail | missing-env | untested), so a failing cell cannot hide behind a
# zero exit from the script. GPU peak is a whole-run delta; per-cell resource
# numbers are only filled where the cell actually measures them.
set -u
BUILD=${1:?build dir}
OUT=${2:?output dir}
M=${MFQ_MODEL:-$HOME/models/Qwen/Qwen3.5-0.8B}
IDS=${MFQ_IDS:-760,6511,314,9338,369,11,321,279,7526,17415,369,279}
BIN=$BUILD/mfq-diagnostics
mkdir -p "$OUT"
[ -x "$BIN" ] || { echo "no $BIN"; exit 1; }
TSV=$OUT/results.tsv
printf 'cell\tstatus\texit\thost_rss_kib\tcommand\tkey\n' >"$TSV"

record() { # record <name> <status> <exit> <rss> <command> <key>
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$5" "$6" >>"$TSV"
    printf '%-26s %-11s exit=%s %s\n' "$1" "$2" "$3" "$6"
}

cell() { # cell <name> <command...>
    name=$1; shift
    cmd=$*
    if [ ! -d "$M" ]; then
        record "$name" missing-env - - "$cmd" "model dir absent: $M"
        return
    fi
    "$@" >"$OUT/$name.log" 2>&1
    code=$?
    rss=-
    [ "$name" = long_decode_rss ] && rss=$(grep -o 'Maximum resident set size.*' "$OUT/$name.log" | grep -o '[0-9]*')
    status=fail
    if [ $code -eq 0 ]; then
        status=pass
    elif grep -qE 'CUDA allocation failed|CUDA VRAM leaves no workspace|cannot load|not found|No such file' "$OUT/$name.log"; then
        status=missing-env
    fi
    key=$(grep -hoE '(prefill_sec|decode_tokens|decode_tok_per_s|ttft_ms|decode_tps|state_check|Maximum resident|PASS|FAIL|cycles|accepted|rejected|graph_captures|speedup|rel|max_abs|text_session_state_check|CUDA allocation|error)[^ ]*.*' "$OUT/$name.log" | head -6 | tr '\n' '|')
    record "$name" "$status" "$code" "$rss" "$cmd" "$key"
}

# Metrics the current CLI cannot produce; recorded so they are not read as passed.
record tail_latency untested - - - "mfq-diagnostics prints no percentile latency"
record logits_baseline untested - - - "mfq-eval-kl needs reference logits artifacts"
record sampling_distribution untested - - - "mfq-diagnostics is fixed greedy (temperature=0, top_k=1)"

# GPU peak is a delta over whatever else is on the card, so record both.
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader >"$OUT/gpu.occupants"
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >"$OUT/gpu.before"
( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 0.3; done ) \
    >"$OUT/gpu.samples" 2>&1 &
SAMPLER=$!

cell greedy            "$BIN" --model "$M" --ids "$IDS" --gen 32 --ctx-size 1024
cell long_decode       "$BIN" --model "$M" --ids "$IDS" --gen 128 --ctx-size 4096
cell long_decode_rss   /usr/bin/time -v "$BIN" --model "$M" --ids "$IDS" --gen 128 --ctx-size 4096
cell prefill_sweep     "$BIN" --model "$M" --ids "$IDS" --prefill-sweep 128,512,1024 --prefill-repeat 3 --ctx-size 1024
cell continuous_batch  "$BIN" --model "$M" --check-continuous-batching --ctx-size 1024
cell session_state     "$BIN" --model "$M" --check-text-session-state --ctx-size 1024
cell mtp_check         "$BIN" --model "$M" --check-qwen35-mtp --ctx-size 1024
cell mtp_bench_mtp     "$BIN" --model "$M" --bench-qwen35-mtp mtp --gen 64 --ctx-size 512
cell mtp_bench_ordinary "$BIN" --model "$M" --bench-qwen35-mtp ordinary --gen 64 --ctx-size 512
cell attention_decode  "$BIN" --model "$M" --check-attention-decode 512 --ctx-size 1024
cell engine_isolation  "$BIN" --model "$M" --check-engine-isolation --ctx-size 1024

kill $SAMPLER 2>/dev/null
BEFORE=$(head -1 "$OUT/gpu.before")
PEAK=$(sort -n "$OUT/gpu.samples" | tail -1)
echo "gpu_used_before_mib=$BEFORE gpu_used_peak_mib=$PEAK gpu_delta_mib=$((PEAK-BEFORE)) (whole run, not per cell)" | tee "$OUT/gpu.peak"

cut -f2 "$TSV" | sort | uniq -c | sed 's/^/  /'
FAILS=$(awk -F'\t' '$2=="fail"' "$TSV" | wc -l)
[ "$FAILS" -eq 0 ] || { echo "$FAILS cell(s) failed; see $TSV"; exit 1; }
