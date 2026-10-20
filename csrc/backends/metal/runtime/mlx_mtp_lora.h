#pragma once

#include "mtp_lora_sessions.h"
#include "mlx_ane_matmul.h"
#include <mlx/mlx.h>
#include <optional>
#include <unordered_map>

namespace mfq::metal {

struct MlxMtpLoraProjection {
    std::string name;
    int input, output, input_offset = 0, output_offset = 0;
};

class MlxMtpLoraParameters {
public:
    MlxMtpLoraParameters(const std::vector<MlxMtpLoraProjection>& layout,
        mlx::core::array a, mlx::core::array b, std::shared_ptr<MlxAneMatmul> engine);
    mlx::core::array linear(const mlx::core::array&, const mlx::core::array&) const;
    mlx::core::array delta(const std::string&, const mlx::core::array&) const;
    mlx::core::Stream cpu() const;
private:
    const std::vector<MlxMtpLoraProjection>& layout_;
    mlx::core::array a_, b_;
    std::shared_ptr<MlxAneMatmul> engine_;
};

using MlxMtpLoraGraph = std::function<mlx::core::array(
    const MlxMtpLoraParameters&, const std::vector<mlx::core::array>&)>;

class MlxMtpLora {
public:
    MlxMtpLora(std::vector<MlxMtpLoraProjection> layout, mlx::core::array frozen_head,
        std::string fingerprint, std::filesystem::path directory,
        std::size_t hot_bytes, std::size_t disk_bytes, int rank = 16, int routes = 10);
    static MlxMtpLora* current() noexcept;
    void begin(const std::string& session);
    void end();
    void begin_round();
    bool collecting() const noexcept { return collect_; }
    mlx::core::array apply(const std::string& projection, const mlx::core::array& input,
        const mlx::core::array& base);
    bool active() const noexcept { return state_ && state_->version > 0; }
    mlx::core::array routed_apply(const std::string& role, const mlx::core::array& input,
        const mlx::core::array& base, const mlx::core::array& ids);
    void prepare_linear(const mlx::core::array& weight);
    void prepare_attention(int dimension, int keys);
    void capture(std::vector<mlx::core::array> constants, MlxMtpLoraGraph);
    void set_final_graph(MlxMtpLoraGraph);
    void verified(const mlx::core::array& teacher_logits, int accepted, double cycle_ms);
    bool close(const std::string& session);
    bool fork(const std::string& source, const std::string& target);
    void clear();
    std::size_t trim(std::size_t bytes);
    std::size_t bytes() const;
    std::vector<std::pair<std::string, double>> metrics() const;
private:
    friend class MlxMtpLoraScope;
    struct Example;
    MtpLoraUpdate train(const MtpLoraState&, const MtpLoraBatch&);
    int inputs_ = 0, outputs_ = 0, vocabulary_, rank_;
    std::size_t training_workspace_ = 0;
    std::vector<MlxMtpLoraProjection> layout_;
    mlx::core::array frozen_head_;
    std::shared_ptr<MlxAneMatmul> engine_;
    std::unique_ptr<MtpLoraSessions> sessions_;
    std::shared_ptr<const MtpLoraState> state_;
    std::optional<mlx::core::array> a_, b_;
    std::unordered_map<std::string, std::pair<mlx::core::array, mlx::core::array>> routed_banks_;
    std::vector<std::shared_ptr<Example>> examples_;
    std::uint64_t device_version_ = UINT64_MAX, budget_skips_ = 0;
    std::uint64_t last_version_ = 0, last_optimizer_step_ = 0;
    bool collect_ = false;
    int depth_ = 0;
};

class MlxMtpLoraScope {
public:
    explicit MlxMtpLoraScope(MlxMtpLora*);
    ~MlxMtpLoraScope();
private:
    MlxMtpLora* previous_;
};

} // namespace mfq::metal
