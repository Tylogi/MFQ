#include "mlx_mtp.h"
#include "grammar_fixture.h"

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

MfqTokenConstraintPtr alternating_constraint() {
    return mfq::metal::test::grammar_constraint(3, "root ::= \"a\" \"b\" root");
}

} // namespace

int main() {
    using mfq::metal::verify_greedy_mtp;
    using mfq::metal::verify_stochastic_mtp;
    using mfq::metal::verify_stochastic_mtp_top_k_device;
    using mfq::metal::verify_stochastic_mtp_top_k_chain_device;
    try {
        {
            bool rejected = false;
            try {
                (void)std::make_shared<MfqTokenConstraint>(nullptr);
            } catch (const std::invalid_argument&) {
                rejected = true;
            }
            auto constraint = alternating_constraint();
            auto cursor = constraint->clone();
            cursor->accept(0);
            if (!rejected || !constraint->allows(0) || constraint->allows(1) ||
                cursor->allows(0) || !cursor->allows(1)) {
                throw std::runtime_error(
                    "MTP token constraint capability mismatch");
            }
        }
        {
            mfq::metal::MlxMtpDepthController controller(3);
            if (controller.depth() != 2) {
                throw std::runtime_error(
                    "adaptive MTP controller did not start at depth two");
            }
            controller.observe(2, 2, 30.0);
            controller.observe(2, 2, 29.0);
            controller.observe(2, 2, 28.0);
            controller.observe(0, 0, 22.0);
            controller.observe(0, 0, 21.0);
            controller.observe(0, 0, 23.0);
            if (controller.depth() < 2 ||
                !controller.measured_cycle_ms(2) ||
                controller.conditional_acceptance(0) <= 0.6) {
                throw std::runtime_error(
                    "adaptive MTP controller warmup mismatch");
            }
        }
        {
            mfq::metal::MlxMtpDepthController controller(1);
            controller.observe(1, 0, 80.0);
            controller.observe(1, 0, 75.0);
            controller.observe(1, 0, 70.0);
            controller.observe(0, 0, 40.0);
            controller.observe(0, 0, 39.0);
            controller.observe(0, 0, 41.0);
            if (controller.depth() != 0 ||
                !controller.measured_cycle_ms(0)) {
                throw std::runtime_error(
                    "depth-one MTP controller skipped plain warmup");
            }
        }
        {
            mfq::metal::MlxMtpDepthController controller(3);
            controller.observe(2, 0, 70.0);
            controller.observe(2, 0, 65.0);
            controller.observe(2, 0, 60.0);
            controller.observe(0, 0, 30.0);
            controller.observe(0, 0, 29.0);
            controller.observe(0, 0, 31.0);
            if (controller.depth() != 0) {
                throw std::runtime_error(
                    "adaptive MTP controller did not select plain decode");
            }
            bool reprobed = false;
            for (int cycle = 0; cycle < 500; ++cycle) {
                const int depth = controller.depth();
                reprobed = reprobed || depth > 0;
                controller.observe(
                    depth,
                    depth,
                    depth > 0 ? 20.0 : 30.0);
            }
            if (!reprobed || controller.depth() == 0) {
                throw std::runtime_error(
                    "adaptive MTP controller did not recover from depth zero");
            }
        }
        {
            mfq::metal::MlxMtpDepthController controller(3, 3);
            controller.observe(3, 3, 45.0);
            controller.observe(3, 3, 44.0);
            controller.observe(3, 3, 43.0);
            controller.observe(0, 0, 30.0);
            controller.observe(0, 0, 30.0);
            controller.observe(0, 0, 30.0);
            if (controller.depth() < 2) {
                throw std::runtime_error(
                    "adaptive MTP controller rejected profitable depth");
            }
        }
        {
            mfq::metal::MlxMtpDepthController controller(5, 5);
            controller.observe(5, 5, 1104.0);
            controller.observe(5, 5, 721.0);
            controller.observe(5, 5, 235.0);
            controller.observe(0, 0, 46.6);
            controller.observe(0, 0, 46.6);
            controller.observe(0, 0, 46.6);
            if (controller.depth() != 5 ||
                controller.conditional_acceptance(4) != 1.0 ||
                !controller.measured_cycle_ms(5) ||
                std::fabs(*controller.measured_cycle_ms(5) - 235.0) > 1e-9) {
                throw std::runtime_error(
                    "adaptive MTP warmup retained cold compile latency");
            }
        }
        {
            mfq::metal::MlxMtpDepthController controller(5);
            controller.observe(2, 2, 30.0);
            controller.observe(2, 2, 29.0);
            controller.observe(2, 2, 28.0);
            controller.observe(0, 0, 20.0);
            controller.observe(0, 0, 20.0);
            controller.observe(0, 0, 20.0);
            for (int cycle = 0;
                 cycle < 100 && controller.depth() != 3;
                 ++cycle) {
                const int depth = controller.depth();
                controller.observe(
                    depth,
                    depth,
                    depth == 0 ? 20.0 : 28.0);
            }
            if (controller.depth() != 3) {
                throw std::runtime_error(
                    "adaptive MTP did not explore an unmeasured depth");
            }
            controller.observe(3, 3, 1000.0);
            if (controller.measured_cycle_ms(3)) {
                throw std::runtime_error(
                    "adaptive MTP exposed the first cold depth sample");
            }
            controller.observe(3, 3, 700.0);
            if (controller.measured_cycle_ms(3)) {
                throw std::runtime_error(
                    "adaptive MTP exposed the second cold depth sample");
            }
            controller.observe(3, 3, 31.0);
            if (!controller.measured_cycle_ms(3) ||
                std::fabs(*controller.measured_cycle_ms(3) - 31.0) > 1e-9) {
                throw std::runtime_error(
                    "adaptive MTP retained cold latency for explored depth");
            }
            controller.observe(3, 3, 30.0);
            if (controller.depth() < 3 ||
                !controller.measured_cycle_ms(3) ||
                *controller.measured_cycle_ms(3) >= 40.0) {
                throw std::runtime_error(
                    "adaptive MTP discouraged a profitable warmed depth");
            }
        }
        const std::array<std::int32_t, 4> drafts{11, 12, 13, 14};
        {
            const std::array<std::int32_t, 5> targets{11, 12, 99, 14, 15};
            const auto result = verify_greedy_mtp(drafts, targets);
            if (result.accepted_drafts != 2 || result.next_token != 99 ||
                result.bonus) {
                throw std::runtime_error("partial MTP acceptance mismatch");
            }
        }
        {
            const std::vector<float> proposal{0.8f, 0.2f};
            const std::vector<float> target{0.4f, 0.6f};
            const std::vector<float> bonus{0.1f, 0.9f};
            const auto accepted = verify_stochastic_mtp(
                0, proposal, target, bonus, 0.25, 0.2);
            if (accepted.accepted_drafts != 1 ||
                accepted.next_token != 1 || !accepted.bonus) {
                throw std::runtime_error(
                    "stochastic MTP acceptance mismatch");
            }
            const auto rejected = verify_stochastic_mtp(
                0, proposal, target, bonus, 0.75, 0.0);
            if (rejected.accepted_drafts != 0 ||
                rejected.next_token != 1 || rejected.bonus) {
                throw std::runtime_error(
                    "stochastic MTP correction mismatch");
            }
        }
        {
            const std::array<std::int32_t, 5> targets{11, 12, 13, 14, 15};
            const auto result = verify_greedy_mtp(drafts, targets);
            if (result.accepted_drafts != 4 || result.next_token != 15 ||
                !result.bonus) {
                throw std::runtime_error("MTP bonus acceptance mismatch");
            }
        }
        {
            auto verification_ids = mfq::metal::mlx_mtp_verification_ids(
                7,
                mlx::core::array({8, 9}, mlx::core::int32),
                2);
            verification_ids.eval();
            const auto* values = verification_ids.data<std::int32_t>();
            if (verification_ids.shape() != mlx::core::Shape{1, 3} ||
                values[0] != 7 || values[1] != 8 || values[2] != 9) {
                throw std::runtime_error(
                    "common MTP verification batch mismatch");
            }

            mlx::core::array verified_hidden(
                {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                mlx::core::Shape{1, 3, 2},
                mlx::core::float32);
            mfq::metal::MlxMtpDraftContext context;
            context.accepted_drafts = 1;
            context.verified_hidden = &verified_hidden;
            auto committed = mfq::metal::mlx_mtp_committed_hidden(
                context, 1, 2);
            committed.eval();
            const auto* hidden = committed.data<float>();
            if (committed.shape() != mlx::core::Shape{1, 2, 2} ||
                hidden[0] != 1.0f || hidden[1] != 2.0f ||
                hidden[2] != 3.0f || hidden[3] != 4.0f) {
                throw std::runtime_error(
                    "common MTP committed hidden prefix mismatch");
            }
        }
        {
            const mlx::core::array proposal_indices(
                {0, 1}, mlx::core::int32);
            const mlx::core::array proposal_probabilities(
                {0.8f, 0.2f}, mlx::core::float32);
            const mlx::core::array target_indices(
                {1, 0}, mlx::core::int32);
            const mlx::core::array target_probabilities(
                {0.6f, 0.4f}, mlx::core::float32);
            const mlx::core::array bonus_indices(
                {1, 0}, mlx::core::int32);
            const mlx::core::array bonus_probabilities(
                {0.9f, 0.1f}, mlx::core::float32);
            const mlx::core::array draft(
                {0}, mlx::core::int32);
            auto accepted = verify_stochastic_mtp_top_k_device(
                proposal_indices,
                proposal_probabilities,
                target_indices,
                target_probabilities,
                bonus_indices,
                bonus_probabilities,
                draft,
                mlx::core::array({0.25f, 0.2f}, mlx::core::float32),
                2);
            accepted.eval();
            const auto* accepted_values =
                accepted.data<std::int32_t>();
            if (accepted_values[0] != 1 || accepted_values[1] != 1 ||
                accepted_values[2] != 0) {
                throw std::runtime_error(
                    "device stochastic MTP acceptance mismatch");
            }

            auto rejected = verify_stochastic_mtp_top_k_device(
                proposal_indices,
                proposal_probabilities,
                target_indices,
                target_probabilities,
                bonus_indices,
                bonus_probabilities,
                draft,
                mlx::core::array({0.75f, 0.0f}, mlx::core::float32),
                2);
            rejected.eval();
            const auto* rejected_values =
                rejected.data<std::int32_t>();
            if (rejected_values[0] != 0 || rejected_values[1] != 1 ||
                rejected_values[2] != 0) {
                throw std::runtime_error(
                    "device stochastic MTP correction mismatch");
            }
        }
        {
            const mlx::core::array proposal_indices(
                {0, 1, 1, 0},
                mlx::core::Shape{2, 2},
                mlx::core::int32);
            const mlx::core::array proposal_probabilities(
                {0.8f, 0.2f, 0.8f, 0.2f},
                mlx::core::Shape{2, 2},
                mlx::core::float32);
            const mlx::core::array target_indices(
                {0, 1, 0, 1, 1, 0},
                mlx::core::Shape{3, 2},
                mlx::core::int32);
            const mlx::core::array target_probabilities(
                {0.9f, 0.1f, 0.9f, 0.1f, 0.7f, 0.3f},
                mlx::core::Shape{3, 2},
                mlx::core::float32);
            auto result = verify_stochastic_mtp_top_k_chain_device(
                proposal_indices,
                proposal_probabilities,
                target_indices,
                target_probabilities,
                mlx::core::array({0, 1}, mlx::core::int32),
                mlx::core::array(
                    {0.5f, 0.5f, 0.25f}, mlx::core::float32),
                2,
                2);
            result.eval();
            const auto* values = result.data<std::int32_t>();
            if (values[0] != 1 || values[1] != 0 ||
                values[2] != 0 || values[3] != 1) {
                throw std::runtime_error(
                    "device stochastic MTP chain mismatch");
            }
        }
        {
            constexpr int top_k = 100;
            std::vector<std::int32_t> proposal_indices(2 * top_k);
            std::vector<std::int32_t> target_indices(3 * top_k);
            std::vector<float> proposal_probabilities(2 * top_k, 0.0f);
            std::vector<float> target_probabilities(3 * top_k, 0.0f);
            for (int row = 0; row < 2; ++row) {
                for (int rank = 0; rank < top_k; ++rank) {
                    proposal_indices[static_cast<std::size_t>(
                        row * top_k + rank)] = rank;
                }
                proposal_probabilities[static_cast<std::size_t>(
                    row * top_k)] = 1.0f;
            }
            for (int row = 0; row < 3; ++row) {
                for (int rank = 0; rank < top_k; ++rank) {
                    target_indices[static_cast<std::size_t>(
                        row * top_k + rank)] = rank;
                }
                target_probabilities[static_cast<std::size_t>(
                    row * top_k)] = 1.0f;
            }
            auto result = verify_stochastic_mtp_top_k_chain_device(
                mlx::core::array(
                    proposal_indices.begin(),
                    mlx::core::Shape{2, top_k}),
                mlx::core::array(
                    proposal_probabilities.begin(),
                    mlx::core::Shape{2, top_k}),
                mlx::core::array(
                    target_indices.begin(),
                    mlx::core::Shape{3, top_k}),
                mlx::core::array(
                    target_probabilities.begin(),
                    mlx::core::Shape{3, top_k}),
                mlx::core::array({0, 0}, mlx::core::int32),
                mlx::core::array(
                    {0.5f, 0.5f, 0.5f}, mlx::core::float32),
                2,
                top_k);
            result.eval();
            const auto* values = result.data<std::int32_t>();
            if (values[0] != 2 || values[1] != 0 ||
                values[2] != 0 || values[3] != 0) {
                throw std::runtime_error(
                    "device stochastic MTP top-k=100 chain mismatch");
            }
        }
        {
            int target_position = 0;
            int resolved_cycles = 0;
            std::vector<std::int64_t> emitted;
            mfq::metal::MlxMtpEngineCallbacks callbacks;
            callbacks.predictor =
                mfq::metal::MlxMtpPredictorDescriptor::recurrent(2);
            callbacks.target_cache_position = [&] {
                return target_position;
            };
            callbacks.prepare_draft = [](
                const mfq::metal::MlxMtpDraftContext& context,
                const mfq::metal::MlxMtpTokenSelector& select_token) {
                for (int position = 0;
                     position < context.requested_depth;
                     ++position) {
                    (void)select_token(mlx::core::array(
                        {10.0f, 0.0f, 0.0f},
                        mlx::core::Shape{1, 3}));
                }
            };
            callbacks.verify_target = [&] (
                std::int32_t,
                const mlx::core::array&,
                int draft_count) {
                target_position += draft_count + 1;
                return mfq::metal::MlxMtpTargetBatch{
                    mlx::core::broadcast_to(
                        mlx::core::array(
                            {10.0f, 0.0f, 0.0f},
                            mlx::core::Shape{1, 3}),
                        mlx::core::Shape{draft_count + 1, 3}),
                    mlx::core::zeros(
                        mlx::core::Shape{1, draft_count + 1, 1}),
                };
            };
            callbacks.resolve_target = [&] (
                int accepted,
                int drafted) {
                target_position -= drafted - accepted;
                ++resolved_cycles;
            };
            mfq::metal::MlxSamplingParams sampling;
            sampling.temperature = 0.0;
            sampling.mtp_max_draft_tokens = 2;
            mfq::metal::MlxMtpGenerationStats stats;
            const auto generated = mfq::metal::run_mlx_mtp_generation(
                mfq::metal::MlxMtpEngineRequest{
                    3,
                    5,
                    32,
                    mlx::core::array(
                        {10.0f, 0.0f, 0.0f},
                        mlx::core::Shape{1, 3}),
                    sampling,
                    std::nullopt,
                    {},
                    [&](std::int64_t token) {
                        emitted.push_back(token);
                        return true;
                    },
                },
                callbacks,
                stats);
            if (generated != 5 || emitted != std::vector<std::int64_t>(5, 0) ||
                target_position != 4 || resolved_cycles != 2 ||
                !stats.available || !stats.used || stats.cycles != 2 ||
                stats.drafted_tokens != 2 || stats.accepted_tokens != 2) {
                throw std::runtime_error(
                    "architecture-independent MTP engine lifecycle mismatch");
            }
        }
        {
            int target_position = 0;
            std::vector<std::int64_t> emitted;
            mfq::metal::MlxMtpEngineCallbacks callbacks;
            callbacks.predictor =
                mfq::metal::MlxMtpPredictorDescriptor::recurrent(2);
            callbacks.target_cache_position = [&] {
                return target_position;
            };
            callbacks.prepare_draft = [](
                const mfq::metal::MlxMtpDraftContext& context,
                const mfq::metal::MlxMtpTokenSelector& select_token) {
                for (int position = 0;
                     position < context.requested_depth;
                     ++position) {
                    (void)select_token(mlx::core::array(
                        {10.0f, 0.0f, 0.0f},
                        mlx::core::Shape{1, 3}));
                }
            };
            callbacks.verify_target = [&] (
                std::int32_t,
                const mlx::core::array&,
                int draft_count) {
                target_position += draft_count + 1;
                return mfq::metal::MlxMtpTargetBatch{
                    mlx::core::broadcast_to(
                        mlx::core::array(
                            {10.0f, 0.0f, 0.0f},
                            mlx::core::Shape{1, 3}),
                        mlx::core::Shape{draft_count + 1, 3}),
                    mlx::core::zeros(
                        mlx::core::Shape{1, draft_count + 1, 1}),
                };
            };
            callbacks.resolve_target = [&] (
                int accepted,
                int drafted) {
                target_position -= drafted - accepted;
            };
            mfq::metal::MlxSamplingParams sampling;
            sampling.temperature = 0.0;
            sampling.mtp_max_draft_tokens = 2;
            mfq::metal::MlxMtpGenerationStats stats;
            const auto generated = mfq::metal::run_mlx_mtp_generation(
                mfq::metal::MlxMtpEngineRequest{
                    3,
                    6,
                    32,
                    mlx::core::array(
                        {10.0f, 0.0f, 0.0f},
                        mlx::core::Shape{1, 3}),
                    sampling,
                    std::nullopt,
                    {},
                    [&](std::int64_t token) {
                        emitted.push_back(token);
                        return true;
                    },
                    0u,
                    alternating_constraint(),
                },
                callbacks,
                stats);
            if (generated != 6 ||
                emitted != std::vector<std::int64_t>({0, 1, 0, 1, 0, 1}) ||
                !stats.available || !stats.used || stats.cycles == 0 ||
                target_position != 5) {
                throw std::runtime_error(
                    "grammar-aware MTP constraint lifecycle mismatch");
            }
        }
        std::cout << "MFQ generic MTP verification tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "MFQ generic MTP verification tests failed: "
                  << error.what() << '\n';
        return 1;
    }
}
