
constexpr char kCsa2Header[] = R"CSA2(
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
inline float v41_fp8(uchar code) {
    const uint a = code & 127, exponent = a >> 3, mantissa = a & 7;
    const float value = exponent == 0 ? ldexp(float(mantissa), -9)
        : ldexp(1.0f + float(mantissa) * 0.125f, int(exponent) - 7);
    return code & 128 ? -value : value;
}
inline float v41_fp4(uchar code) {
    const float table[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    return (code & 8 ? -1.0f : 1.0f) * table[code & 7];
}
)CSA2";

constexpr char kCsa2Attention[] = R"CSA2(
    const uint lane = thread_index_in_threadgroup % 32;
    const uint head = threadgroup_position_in_grid.x * 4
        + thread_index_in_threadgroup / 32;
    const uint query = threadgroup_position_in_grid.y;
    const uint split = threadgroup_position_in_grid.z;
    const int H = meta[0], W = meta[1], C = meta[2];
    const int NW = meta[3], NC = meta[4], NS = meta[5];
    if (head >= H) return;
    // Contiguous lanes stay inside one quantization group at these widths.
    constexpr bool REUSE_SCALE = D == 32 || D == 64 || D == 128 || D == 256 || D == 512;
    float query_values[D / 32], acc[D / 32];
    for (int v = 0; v < D / 32; ++v) {
        query_values[v] = float(q[(query * H + head) * D + (REUSE_SCALE ? lane * (D / 32) + v : lane + v * 32)]);
        acc[v] = 0.0f;
    }
    float maximum = -INFINITY, denominator = 0.0f;
    const int end = min(int((split + 1) * CHUNK), W + C);
    for (int j = split * CHUNK; j < end; ++j) {
        const bool compressed = j >= W;
        const int row = compressed ? ci[query * C + j - W] : wi[query * W + j];
        if (row < 0 || row >= (compressed ? NC : NW)) continue;
        const size_t base = size_t(row) * (compressed
            ? (D / 2 + D / 16) : (D + D / 32));
        const uint first = lane * (D / 32);
        const float pooled_scale = REUSE_SCALE && compressed
            ? v41_fp8(pooled[base + D / 2 + first / 16]) : 0.0f;
        const int window_exp = REUSE_SCALE && !compressed
            ? int(window[base + D + first / 32]) - 127 : 0;
        float values[D / 32], dot = 0.0f;
        for (int v = 0; v < D / 32; ++v) {
            const uint d = (REUSE_SCALE ? lane * (D / 32) + v : lane + v * 32);
            float value;
            if (compressed) {
                const uchar code = (pooled[base + d / 2] >> ((d % 2) * 4)) & 15;
                value = v41_fp4(code) * (REUSE_SCALE ? pooled_scale : v41_fp8(pooled[base + D / 2 + d / 16]));
            } else {
                value = ldexp(v41_fp8(window[base + d]), REUSE_SCALE ? window_exp : int(window[base + D + d / 32]) - 127);
            }
            values[v] = value;
            dot += query_values[v] * value;
        }
        const float score = simd_sum(dot) * scalep[0];
        const float next_maximum = max(maximum, score);
        const float correction = exp(maximum - next_maximum);
        const float probability = exp(score - next_maximum);
        denominator = denominator * correction + probability;
        for (int v = 0; v < D / 32; ++v)
            acc[v] = acc[v] * correction + probability * values[v];
        maximum = next_maximum;
    }
    const size_t base = ((size_t(query) * H + head) * NS + split) * (D + 2);
    for (int v = 0; v < D / 32; ++v) partial[base + (REUSE_SCALE ? lane * (D / 32) + v : lane + v * 32)] = acc[v];
    if (lane == 0) {
        partial[base + D] = maximum;
        partial[base + D + 1] = denominator;
    }
)CSA2";

constexpr char kCsa2AttentionMma[] = R"CSA2(
    const uint tid = thread_index_in_threadgroup, lane = tid % 32, sg = tid / 32;
    const uint query = threadgroup_position_in_grid.y, split = threadgroup_position_in_grid.z;
    const uint first_head = threadgroup_position_in_grid.x * 8;
    const int H = meta[0], W = meta[1], C = meta[2], NW = meta[3], NC = meta[4], NS = meta[5];
    const uint fm = ((lane >> 2) & 4) + ((lane >> 1) & 3);
    const uint fn = (((lane >> 2) & 2) << 1) + ((lane & 1) << 1);
    threadgroup float tile[8 * D], dots[4 * 64], probability[64];
    threadgroup float maxima[8], denominator[8], correction[8];
    threadgroup bool valid[8];
    simdgroup_matrix<float, 8, 8> output[D / 32];
    for (int v = 0; v < D / 32; ++v) {
        output[v].thread_elements()[0] = 0.0f;
        output[v].thread_elements()[1] = 0.0f;
    }
    if (tid < 8) { maxima[tid] = -INFINITY; denominator[tid] = 0.0f; }
    const int end = min(int((split + 1) * CHUNK), W + C);
    for (int begin = split * CHUNK; begin < end; begin += 8) {
        for (uint i = tid; i < 8 * D; i += 128) {
            const uint r = i / D, d = i % D, j = begin + r;
            const bool compressed = j >= W;
            int row = -1;
            if (j < end) row = compressed ? ci[query * C + j - W] : wi[query * W + j];
            const bool ok = row >= 0 && row < (compressed ? NC : NW);
            float value = 0.0f;
            if (ok) {
                if (compressed) {
                    const size_t base = size_t(row) * (D / 2 + D / 16);
                    const uchar code = (pooled[base + d / 2] >> ((d % 2) * 4)) & 15;
                    value = v41_fp4(code) * v41_fp8(pooled[base + D / 2 + d / 16]);
                } else {
                    const size_t base = size_t(row) * (D + D / 32);
                    value = ldexp(v41_fp8(window[base + d]), int(window[base + D + d / 32]) - 127);
                }
            }
            tile[i] = value;
            if (d == 0) valid[r] = ok;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Four simdgroups split the QK dot product's channel dimension.
        simdgroup_matrix<float, 8, 8> score, a, b;
        score.thread_elements()[0] = 0.0f;
        score.thread_elements()[1] = 0.0f;
        for (int k = sg * (D / 4); k < (sg + 1) * (D / 4); k += 8) {
            const uint head = first_head + fm;
            a.thread_elements()[0] = head < H ? float(q[(query * H + head) * D + k + fn]) : 0.0f;
            a.thread_elements()[1] = head < H ? float(q[(query * H + head) * D + k + fn + 1]) : 0.0f;
            b.thread_elements()[0] = tile[fn * D + k + fm];
            b.thread_elements()[1] = tile[(fn + 1) * D + k + fm];
            simdgroup_multiply_accumulate(score, a, b, score);
        }
        dots[sg * 64 + fm * 8 + fn] = score.thread_elements()[0];
        dots[sg * 64 + fm * 8 + fn + 1] = score.thread_elements()[1];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 8) {
            float values[8], next_max = maxima[tid];
            for (uint r = 0; r < 8; ++r) {
                float sum = dots[tid * 8 + r] + dots[64 + tid * 8 + r]
                    + dots[128 + tid * 8 + r] + dots[192 + tid * 8 + r];
                values[r] = valid[r] ? sum * scalep[0] : -INFINITY;
                next_max = max(next_max, values[r]);
            }
            correction[tid] = maxima[tid] == -INFINITY ? 0.0f : exp(maxima[tid] - next_max);
            float total = denominator[tid] * correction[tid];
            for (uint r = 0; r < 8; ++r) {
                const float prob = valid[r] ? exp(values[r] - next_max) : 0.0f;
                probability[tid * 8 + r] = prob;
                total += prob;
            }
            maxima[tid] = next_max;
            denominator[tid] = total;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // PV reuses the decoded tile. Float matrices retain FP32 probabilities.
        a.thread_elements()[0] = probability[fm * 8 + fn];
        a.thread_elements()[1] = probability[fm * 8 + fn + 1];
        for (int v = 0; v < D / 32; ++v) {
            const uint d = sg * (D / 4) + v * 8 + fn;
            b.thread_elements()[0] = tile[fm * D + d];
            b.thread_elements()[1] = tile[fm * D + d + 1];
            output[v].thread_elements()[0] *= correction[fm];
            output[v].thread_elements()[1] *= correction[fm];
            simdgroup_multiply_accumulate(output[v], a, b, output[v]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint head = first_head + fm;
    if (head < H) {
        const size_t base = ((size_t(query) * H + head) * NS + split) * (D + 2);
        for (int v = 0; v < D / 32; ++v) {
            const uint d = sg * (D / 4) + v * 8 + fn;
            partial[base + d] = output[v].thread_elements()[0];
            partial[base + d + 1] = output[v].thread_elements()[1];
        }
    }
    if (tid < 8 && first_head + tid < H) {
        const size_t base = ((size_t(query) * H + first_head + tid) * NS + split) * (D + 2);
        partial[base + D] = maxima[tid];
        partial[base + D + 1] = denominator[tid];
    }
)CSA2";

constexpr char kCsa2Index[] = R"CSA2(
    const uint lane = thread_index_in_threadgroup % 32;
    const uint position = threadgroup_position_in_grid.x * 4
        + thread_index_in_threadgroup / 32;
    const uint query = threadgroup_position_in_grid.y;
    const int H = HEADS, N = meta[1], M = meta[2], start = meta[3], ratio = RATIO;
    if (position >= M) return;
    const int row = CANDIDATES ? candidates[query * M + position] : int(position);
    if (row < 0 || row >= N || row + meta[5] >= (start + int(query) + 1) / ratio) {
        if (lane == 0) scores[query * M + position] = -INFINITY;
        return;
    }
    const size_t base = size_t(row) * (D / 2 + D / 32);
    float key[D / 32];
    for (int v = 0; v < D / 32; ++v) {
        const uint d = lane + v * 32;
        const uchar code = (keys[base + d / 2] >> ((d % 2) * 4)) & 15;
        key[v] = ldexp(v41_fp4(code), int(keys[base + D / 2 + d / 32]) - 127);
    }
    float result = 0.0f;
    for (int h = 0; h < H; ++h) {
        float dot = 0.0f;
        for (int v = 0; v < D / 32; ++v)
            dot += float(q[(query * H + h) * D + lane + v * 32]) * key[v];
        result += max(simd_sum(dot), 0.0f) * weights[query * H + h];
    }
    if (lane == 0) scores[query * M + position] = result;
)CSA2";

constexpr char kCsa2IndexMma[] = R"CSA2(
    const uint tid = thread_index_in_threadgroup;
    const uint lane = tid % 32, sg = tid / 32;
    const uint query = threadgroup_position_in_grid.y;
    constexpr uint COLS = HEAD_SPLIT ? 16 : 32;
    const uint key_group = HEAD_SPLIT ? sg % 2 : sg;
    const uint block = threadgroup_position_in_grid.x * COLS;
    const int H = HEADS, N = meta[1], M = meta[2], start = meta[3], ratio = RATIO;
    const uint fm = ((lane >> 2) & 4) + ((lane >> 1) & 3);
    const uint fn = (((lane >> 2) & 2) << 1) + ((lane & 1) << 1);
    threadgroup float tile[D * COLS];
    for (uint i = tid; i < D * COLS; i += 128) {
        const uint d = i / COLS, col = i % COLS, pos = block + col;
        const int row = pos < M ? (CANDIDATES ? candidates[query * M + pos] : int(pos)) : -1;
        float value = 0.0f;
        if (row >= 0 && row < N && row + meta[5] < (start + int(query) + 1) / ratio) {
            const size_t base = size_t(row) * (D / 2 + D / 32);
            const uchar code = (keys[base + d / 2] >> ((d % 2) * 4)) & 15;
            value = ldexp(v41_fp4(code), int(keys[base + D / 2 + d / 32]) - 127);
        }
        tile[i] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total0 = 0.0f, total1 = 0.0f;
    for (int h = HEAD_SPLIT ? int(sg / 2) * 8 : 0; h < H; h += HEAD_SPLIT ? 16 : 8) {
        simdgroup_matrix<float, 8, 8> acc, a, b;
        acc.thread_elements()[0] = 0.0f;
        acc.thread_elements()[1] = 0.0f;
        #pragma clang loop unroll(full)
        for (int d = 0; d < D; d += 8) {
            a.thread_elements()[0] = h + fm < H ? float(q[(query * H + h + fm) * D + d + fn]) : 0.0f;
            a.thread_elements()[1] = h + fm < H ? float(q[(query * H + h + fm) * D + d + fn + 1]) : 0.0f;
            b.thread_elements()[0] = tile[(d + fm) * COLS + key_group * 8 + fn];
            b.thread_elements()[1] = tile[(d + fm) * COLS + key_group * 8 + fn + 1];
            simdgroup_multiply_accumulate(acc, a, b, acc);
        }
        const float weight = h + fm < H ? weights[query * H + h + fm] : 0.0f;
        float v0 = max(acc.thread_elements()[0], 0.0f) * weight;
        float v1 = max(acc.thread_elements()[1], 0.0f) * weight;
        // Lane bits 1, 2 and 4 span the eight head rows of each column.
        for (uint i = 0; i < 3; ++i) {
            const uint mask = i == 0 ? 2 : (i == 1 ? 4 : 16);
            v0 += simd_shuffle_xor(v0, mask);
            v1 += simd_shuffle_xor(v1, mask);
        }
        total0 += v0;
        total1 += v1;
    }
    threadgroup float head_totals[32];
    if (HEAD_SPLIT) {
        if (fm == 0) {
            head_totals[sg * 8 + fn] = total0;
            head_totals[sg * 8 + fn + 1] = total1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < COLS && block + tid < M) {
            const uint pos = block + tid;
            const int row = CANDIDATES ? candidates[query * M + pos] : int(pos);
            const bool valid = row >= 0 && row < N && row + meta[5] < (start + int(query) + 1) / ratio;
            scores[query * M + pos] = valid ? head_totals[tid] + head_totals[tid + COLS] : -INFINITY;
        }
    } else if (fm == 0) {
        for (uint j = 0; j < 2; ++j) {
            const uint pos = block + sg * 8 + fn + j;
            if (pos >= M) continue;
            const int row = CANDIDATES ? candidates[query * M + pos] : int(pos);
            const bool valid = row >= 0 && row < N && row + meta[5] < (start + int(query) + 1) / ratio;
            scores[query * M + pos] = valid ? (j == 0 ? total0 : total1) : -INFINITY;
        }
    }
)CSA2";

constexpr char kCsa2Merge[] = R"CSA2(
    const uint lane = thread_index_in_threadgroup % 32;
    const uint head = threadgroup_position_in_grid.x * 4
        + thread_index_in_threadgroup / 32;
    const uint query = threadgroup_position_in_grid.y;
    const int H = meta[0], NS = meta[1];
    if (head >= H) return;
    const size_t base = (size_t(query) * H + head) * NS * (D + 2);
    float maximum = float(sink[head]);
    for (int s = 0; s < NS; ++s)
        maximum = max(maximum, partial[base + s * (D + 2) + D]);
    float denominator = exp(float(sink[head]) - maximum);
    float acc[D / 32];
    for (int v = 0; v < D / 32; ++v) acc[v] = 0.0f;
    for (int s = 0; s < NS; ++s) {
        const size_t pos = base + s * (D + 2);
        const float correction = exp(partial[pos + D] - maximum);
        denominator += partial[pos + D + 1] * correction;
        for (int v = 0; v < D / 32; ++v)
            acc[v] += partial[pos + lane + v * 32] * correction;
    }
    for (int v = 0; v < D / 32; ++v)
        out[(query * H + head) * D + lane + v * 32] = T(acc[v] / denominator);
)CSA2";

constexpr char kCsa2Topk[] = R"CSA2(
    const uint tid = thread_index_in_threadgroup;
    const uint query = threadgroup_position_in_grid.y;
    const uint tile = threadgroup_position_in_grid.x;
    const int width = meta[0], base = meta[1];
    // Runtime counts keep one pipeline as the causal key count grows.
    const uint K = uint(meta[5]), NT = uint(meta[6]);
    threadgroup float exchange_values[TILE];
    threadgroup int exchange_ids[TILE];
    float values[TILE / THREADS], next_values[TILE / THREADS];
    int indices[TILE / THREADS], next_ids[TILE / THREADS];
    for (uint j = 0; j < TILE / THREADS; ++j) {
        const uint i = tid + j * THREADS, pos = tile * TILE + i;
        float value = -INFINITY;
        if (pos < width) {
            for (uint k = 0; k < BLOCK; ++k) {
                const uint src = pos * BLOCK + k;
                if (src < uint(meta[2])) value = max(value, scores[query * meta[2] + src]);
            }
            const int length = (meta[3] + int(query) + 1) / RATIO;
            if (FORCE && length > 0 && int(pos) + base == (length - 1) / BLOCK)
                value = INFINITY;
        }
        values[j] = value;
        indices[j] = pos < width ? (EXPLICIT ? ids[query * width + pos] : base + int(pos)) : INT_MAX;
    }
    // Only cross-SIMD exchanges need threadgroup storage and barriers.
    for (uint size = 2; size <= TILE; size <<= 1) {
        for (uint stride = size >> 1; stride; stride >>= 1) {
            const bool cross_simd = stride >= 32 && stride < THREADS;
            if (cross_simd) {
                for (uint j = 0; j < TILE / THREADS; ++j) {
                    exchange_values[tid + j * THREADS] = values[j];
                    exchange_ids[tid + j * THREADS] = indices[j];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            for (uint j = 0; j < TILE / THREADS; ++j) {
                const uint i = tid + j * THREADS;
                float other;
                int other_id;
                if (stride < 32) {
                    other = simd_shuffle_xor(values[j], stride);
                    other_id = simd_shuffle_xor(indices[j], stride);
                } else if (cross_simd) {
                    other = exchange_values[i ^ stride];
                    other_id = exchange_ids[i ^ stride];
                } else {
                    other = values[j ^ (stride / THREADS)];
                    other_id = indices[j ^ (stride / THREADS)];
                }
                const bool better = other > values[j] || (other == values[j] && other_id < indices[j]);
                const bool worse = other < values[j] || (other == values[j] && other_id > indices[j]);
                const bool descending = ((i & size) == 0) == ((i & stride) == 0);
                const bool take = descending ? better : worse;
                next_values[j] = take ? other : values[j];
                next_ids[j] = take ? other_id : indices[j];
            }
            if (cross_simd) threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint j = 0; j < TILE / THREADS; ++j) {
                values[j] = next_values[j]; indices[j] = next_ids[j];
            }
        }
    }
    for (uint j = 0; j < TILE / THREADS; ++j) {
        const uint i = tid + j * THREADS;
        if (i < K) {
            const size_t out = (size_t(query) * NT + tile) * K + i;
            selected_scores[out] = values[j];
            selected_ids[out] = indices[j];
        }
    }

)CSA2";

constexpr char kCsa2TopkMerge[] = R"CSA2(
    const uint pos = thread_position_in_grid.x;
    const uint query = threadgroup_position_in_grid.y;
    const uint group = threadgroup_position_in_grid.z;
    const uint na = meta[0], nb = meta[1], runs = meta[2], K = meta[3];
    const uint groups = GROUPED ? (runs + FANIN - 1) / FANIN : 1;
    if (GROUPED) {
        const uint source = pos / K, own = pos % K;
        const uint first = group * FANIN, n = min(uint(FANIN), runs - first);
        if (source >= n) return;
        const size_t base = (size_t(query) * runs + first) * K;
        const float value = a_scores[base + source * K + own];
        const int id = a_ids[base + source * K + own];
        uint rank = own;
        for (uint run = 0; run < n; ++run) {
            if (run == source) continue;
            uint lo = 0, hi = K - rank;
            while (lo < hi) {
                const uint mid = (lo + hi) / 2;
                const float other = a_scores[base + run * K + mid];
                const int other_id = a_ids[base + run * K + mid];
                const bool before = other > value || (other == value &&
                    (other_id < id || (run < source && other_id == id)));
                if (before) lo = mid + 1; else hi = mid;
            }
            rank += lo;
            // Once K entries precede this item, later runs cannot rescue it.
            if (rank >= K) return;
        }
        const size_t out = (size_t(query) * groups + group) * K + rank;
        selected_scores[out] = value;
        selected_ids[out] = id;
        return;
    }
    const uint left_n = na, right_n = nb;
    if (pos >= left_n + right_n) return;
    const size_t abase = size_t(query) * na, bbase = size_t(query) * nb;
    const bool right = pos >= left_n;
    const uint own = right ? pos - left_n : pos;
    const float value = right ? b_scores[bbase + own] : a_scores[abase + own];
    const int id = right ? b_ids[bbase + own] : a_ids[abase + own];
    if (own >= K) return;
    uint lo = 0, hi = min(right ? left_n : right_n, uint(K) - own);
    while (lo < hi) {
        const uint mid = (lo + hi) / 2;
        const float other = right ? a_scores[abase + mid] : b_scores[bbase + mid];
        const int other_id = right ? a_ids[abase + mid] : b_ids[bbase + mid];
        // Identical entries in the left run precede those in the right run.
        const bool before = other > value || (other == value &&
            (other_id < id || (right && other_id == id)));
        if (before) lo = mid + 1; else hi = mid;
    }
    const uint rank = own + lo;
    if (rank < K) {
        const size_t out = (size_t(query) * groups + group) * K + rank;
        selected_scores[out] = value;
        selected_ids[out] = id;
    }
)CSA2";

constexpr char kCsa2RadixHeader[] = R"CSA2(
// Order-preserving key: larger scores give larger keys, and -0 equals +0.
inline uint v41_score_key(float x) {
    const uint u = as_type<uint>(x == 0.0f ? 0.0f : x);
    return (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
}
)CSA2";

constexpr char kCsa2RadixTopk[] = R"CSA2(
    constexpr uint CAP = 2048, BINS = 4096;
    const uint tid = thread_index_in_threadgroup;
    const uint row = threadgroup_position_in_grid.x;
    const uint n = scores_shape[2], K = meta[0];
    const device float* s = scores + size_t(row) * n;
    device int* out = ids + size_t(row) * K;
    threadgroup ulong buf[CAP];
    threadgroup atomic_uint* hist = (threadgroup atomic_uint*)buf;
    threadgroup atomic_uint counters[2];
    threadgroup uint part[256];
    threadgroup uint threshold_bin, above, need;
    for (uint b = tid; b < BINS; b += 256)
        atomic_store_explicit(hist + b, 0, memory_order_relaxed);
    if (tid < 2) atomic_store_explicit(counters + tid, 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < n; i += 256)
        atomic_fetch_add_explicit(
            hist + (v41_score_key(s[i]) >> 20), 1, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Thread t owns 16 bins counted down from the top bin.
    uint local = 0;
    for (uint j = 0; j < 16; ++j)
        local += atomic_load_explicit(
            hist + (BINS - 1 - (tid * 16 + j)), memory_order_relaxed);
    part[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint acc = 0, t = 0;
        for (; t < 255; ++t) {
            if (acc + part[t] >= K) break;
            acc += part[t];
        }
        for (uint j = 0; j < 16; ++j) {
            const uint bin = BINS - 1 - (t * 16 + j);
            const uint c = atomic_load_explicit(hist + bin, memory_order_relaxed);
            if (acc + c >= K || j == 15) {
                threshold_bin = bin;
                above = acc;
                need = K - acc;
                break;
            }
            acc += c;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint bin = threshold_bin;
    for (uint i = tid; i < n; i += 256) {
        const uint key = v41_score_key(s[i]), high = key >> 20;
        if (high > bin) {
            out[atomic_fetch_add_explicit(counters, 1, memory_order_relaxed)] = int(i);
        } else if (high == bin) {
            const uint c = atomic_fetch_add_explicit(counters + 1, 1, memory_order_relaxed);
            if (c < CAP) buf[c] = (ulong(key) << 32) | ulong(0xFFFFFFFFu - i);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    const uint count = atomic_load_explicit(counters + 1, memory_order_relaxed);
    if (count <= CAP) {
        uint size = 1;
        while (size < count) size <<= 1;
        for (uint i = count + tid; i < size; i += 256) buf[i] = 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Bitonic sort, descending by key, then ascending by id.
        for (uint k = 2; k <= size; k <<= 1) {
            for (uint j = k >> 1; j > 0; j >>= 1) {
                for (uint i = tid; i < size; i += 256) {
                    const uint l = i ^ j;
                    if (l > i) {
                        const ulong a = buf[i], b = buf[l];
                        if ((i & k) == 0 ? a < b : a > b) {
                            buf[i] = b;
                            buf[l] = a;
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        for (uint r = tid; r < need; r += 256)
            out[above + r] = int(0xFFFFFFFFu - uint(buf[r] & 0xFFFFFFFFul));
        return;
    }
    threadgroup uint prefix, remaining, greater[256], equal[256];
    if (tid == 0) {
        prefix = 0;
        remaining = K;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int shift = 24; shift >= 0; shift -= 8) {
        atomic_store_explicit(hist + tid, 0, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < n; i += 256) {
            const uint key = v41_score_key(s[i]);
            if (shift == 24 || (key >> (shift + 8)) == prefix)
                atomic_fetch_add_explicit(
                    hist + ((key >> shift) & 255u), 1, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            for (int b = 255; b >= 0; --b) {
                const uint c = atomic_load_explicit(hist + b, memory_order_relaxed);
                if (remaining > c) {
                    remaining -= c;
                } else {
                    prefix = (prefix << 8) | uint(b);
                    break;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // prefix is now the K-th largest key. Write greater keys, then equal keys
    // by ascending id, using per-thread ranges for a stable order.
    const uint first = (n * tid) / 256, last = (n * (tid + 1)) / 256;
    uint ng = 0, ne = 0;
    for (uint i = first; i < last; ++i) {
        const uint key = v41_score_key(s[i]);
        ng += key > prefix;
        ne += key == prefix;
    }
    greater[tid] = ng;
    equal[tid] = ne;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint g = 0, e = 0;
        for (uint t = 0; t < 256; ++t) {
            const uint a = greater[t], b = equal[t];
            greater[t] = g;
            equal[t] = e;
            g += a;
            e += b;
        }
        remaining = g;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint g = greater[tid], e = equal[tid];
    for (uint i = first; i < last; ++i) {
        const uint key = v41_score_key(s[i]);
        if (key > prefix) {
            out[g++] = int(i);
        } else if (key == prefix) {
            if (remaining + e < K) out[remaining + e] = int(i);
            ++e;
        }
    }
)CSA2";
