#include "engine/batch_execution.h"
#include "request_executor.h"
#include "common/causal_metadata.h"
#include "storage/state/batch_paged.h"

#include <array>
#include <iostream>
#include <map>

using namespace mfq::cuda;
using namespace mfq::engine;

static void check(bool valid, const char* message) { MFQ_RUNTIME_CHECK(valid, message); }

// Independent request histories, not stable slots or Qwen layer-state bundles.
// Instantiating the CUDA runner here rejects a hidden StateAdapter protocol.
struct DenseRequestState { int64_t position = 0; bool active = false; };
struct DenseModel;
struct DenseStorage {
    using Row = std::pair<DenseRequestState*, int64_t>;
    std::map<DenseRequestState*, int64_t> histories;
    DenseStorage(DenseModel&, size_t, bool) {}
    ~DenseStorage() { check(histories.empty(), "dense histories leaked"); }
    struct Prefill {
        DenseStorage& owner;
        DenseRequestState& request;
        int64_t end, tokens;
        void commit() { request.position = end; request.active = end == tokens; owner.histories[&request] = end; }
    };
    Prefill prefill(DenseRequestState& request, int64_t offset, int64_t count, int64_t tokens) {
        check(request.position == offset, "dense prefill position changed");
        return {*this, request, offset + count, tokens};
    }
    struct Decode {
        DenseStorage& owner;
        std::span<const Row> rows;
        void commit() {
            for (const auto& [request, position] : rows) owner.histories[request] = ++request->position;
        }
        void abort() {} // No numerical history is mutated before commit.
    };
    Decode decode(std::span<const Row> rows, int64_t batch) {
        check(rows.size() <= batch, "dense physical rows exceeded capacity");
        for (const auto& [request, position] : rows)
            check(request->active && histories.at(request) == position, "dense decode lost history");
        return {*this, rows};
    }
    void release(DenseRequestState& request) { histories.erase(&request); request = {}; }
    void recover(std::span<DenseRequestState* const> requests) { for (auto* request : requests) release(*request); }
    BatchStorageStats stats() const {
        return {int64_t(std::count_if(histories.begin(), histories.end(), [](const auto& row) {
            return row.first->active;
        })), 0, std::nullopt};
    }
};
struct DenseModel {
    using BatchRequestState = DenseRequestState;
    using BatchStorage = DenseStorage;
    CudaExecutionContext* execution;
    mfq::models::CausalLmMetadata metadata;
    std::vector<std::unique_ptr<Block>> blocks;
    int64_t cache_pos = 0;
    explicit DenseModel(CudaExecutionContext& execution) : execution(&execution) {
        metadata.decode_graph_double_warmup = true;
    }
    int64_t vocab_size() const { return 32; }
    int64_t max_position_embeddings() const { return 256; }
    int64_t num_experts() const { return 0; }
    bool uses_decode_sequence_length() const { return false; }
    std::vector<const void*> batch_state_addresses(const auto&) const { return {}; }
    Tensor hidden_forward(Tensor ids, MfqOptional<Tensor> = {}, MfqOptional<Tensor> = {},
                          void* = nullptr, MfqOptional<Tensor> = {}) {
        return ids.to(kFloat32).unsqueeze(-1);
    }
    struct Plan { int64_t length, parts; };
    Tensor hidden_forward_static(Tensor ids, Tensor, Tensor, Plan) { return hidden_forward(ids); }
    Tensor logits_from_hidden(Tensor hidden) {
        const auto next = hidden.to(kFloat32).reshape({hidden.size(0), 1}) + 1;
        return -(mfq_tensor_backend::arange(32, next.options()).reshape({1, 32}) - next).square();
    }
};

static void check_dense_execution(bool graph) {
    CudaExecutionContext context;
    CudaExecutionScope scope(context);
    DenseModel model(context);
    CudaContinuousBatchConfig config;
    config.max_sequences = 4; config.cuda_graph = graph; config.cuda_graph_minimum_tokens = 2;
    struct Ops {
        ContinuousBatch<CudaBatchExecution<DenseModel>> batch;
        Ops(DenseModel& model, CudaExecutionContext& context, CudaContinuousBatchConfig config)
            : batch(2, 2, model, context, config) {}
        bool can_overlap(const EngineRequest&) const { return true; }
        bool exclusive() const { return false; }
        bool mtp_available() const { return false; }
        void execute(const std::vector<RequestId>& eligible) { batch.step(eligible); }
        Generation generate(const RequestId& id, ExecutionRequest& request) {
            return batch.generate(id, request.input, request.output);
        }
    } ops(model, context, config);
    RequestExecutor executor(4);
    EngineInfo info; info.max_requests = 4; info.max_context = 256; info.vocab_size = 32;
    for (int turn = 0; turn < 8; ++turn) {
        std::map<std::string, std::vector<int64_t>> tokens;
        int terminals = 0, paused = 0;
        for (const auto& id : {"a", "b", "cancel"}) {
            EngineRequest request; request.id = id;
            request.diagnostic_input = DiagnosticTokenInput{{1, 2, 3}};
            request.input.sampling.max_tokens = 6; request.input.sampling.temperature = 0;
            check(executor.admit(std::move(request), nullptr, info, ops) == Admission::accepted, "dense admission failed");
        }
        executor.cancel("cancel");
        for (int tick = 0; tick < 100 && !executor.empty(); ++tick) {
            const bool pause = !tokens["a"].empty() && paused < 4;
            const auto before = tokens["a"].size();
            const auto result = executor.step(pause ? std::vector<RequestId>{"b"} : std::vector<RequestId>{"a", "b"}, ops);
            check(result.status.healthy, "dense CUDA quantum failed");
            for (const auto& event : result.events) {
                if (const auto* delta = std::get_if<OutputDelta>(&event.data))
                    tokens[event.id].insert(tokens[event.id].end(), delta->token_ids.begin(), delta->token_ids.end());
                terminals += terminal(event.data);
            }
            if (pause) { ++paused; check(tokens["a"].size() == before, "paused dense row advanced"); }
        }
        check(executor.empty() && terminals == 3 && paused == 4, "dense requests did not retire exactly once");
        check(tokens["a"] == std::vector<int64_t>({4, 5, 6, 7, 8, 9}) && tokens["a"] == tokens["b"] &&
              tokens["cancel"].empty(), "dense batched output differs from numerical oracle");
    }
    for (const auto& [key, value] : ops.batch.metrics()) {
        if (key == "continuous_batching_active") check(value == 0, "dense storage remained active");
        if (graph && key == "continuous_batching_cuda_graph_prewarms") check(value > 0, "dense Graph path was not exercised");
    }
}

static void check_paged_storage() {
    const std::array<PagedKvLayerGeometry, 2> geometry{{{0, 1, 4}, {0, 2, 3}}};
    BatchPagedKvPool pool(geometry, 8, 80);
    PagedKvSequence first, second, padding;
    pool.ensure_tokens(first, 17); pool.ensure_tokens(second, 33); pool.ensure_tokens(padding, 1);
    const std::array<const PagedKvSequence*, 3> rows{{&second, &padding, &first}};
    pool.bind(rows);
    const auto view = pool.view(1);
    check(view.batch_size() == 3 && view.paged_heads == 2 && view.paged_head_dim == 3, "paged view ignored native geometry");
    auto table = view.page_table.cpu().contiguous();
    const auto* data = table.data_ptr<int32_t>();
    check(data[0] == second.physical_pages[0] && data[5] == padding.physical_pages[0] &&
          data[10] == first.physical_pages[0] && data[6] == -1, "paged row mapping or padding changed");
    const auto retained = pool.stats().reserved_bytes;
    pool.release(first); pool.release(second); pool.release(padding);
    for (int turn = 0; turn < 64; ++turn) {
        pool.ensure_tokens(first, 16 + turn % 64);
        pool.release(first);
        check(pool.stats().live_pages == 0 && pool.stats().reserved_bytes == retained, "paged churn grew retained storage");
    }
    bool rejected = false;
    try { pool.bind(rows); } catch (const std::exception&) { rejected = true; }
    // Empty page lists are valid; an explicitly stale page is not.
    check(!rejected, "empty owned sequences were rejected");
    first.physical_pages.push_back(0);
    try { pool.bind(rows); } catch (const std::exception&) { rejected = true; }
    check(rejected, "paged pool accepted an unowned page");
}

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || !count) return 77;
    try {
        check_paged_storage();
        check_dense_execution(false);
        check_dense_execution(true);
        std::cout << "CUDA batch execution/storage checks passed\n";
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
