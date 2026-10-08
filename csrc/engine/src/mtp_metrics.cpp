#include "mtp_metrics.h"

namespace mfq::engine::mtp {

void append_generation_metrics(
        std::vector<std::pair<std::string, double>>& metrics,
        const GenerationStats& stats) {
    metrics.emplace_back("mtp_used", stats.used ? 1.0 : 0.0);
    metrics.emplace_back("mtp_cycles", static_cast<double>(stats.cycles));
    metrics.emplace_back(
        "mtp_drafted_tokens", static_cast<double>(stats.drafted_tokens));
    metrics.emplace_back(
        "mtp_accepted_tokens", static_cast<double>(stats.accepted_tokens));
    metrics.emplace_back(
        "mtp_acceptance_rate",
        stats.drafted_tokens == 0
            ? 0.0
            : static_cast<double>(stats.accepted_tokens) /
                stats.drafted_tokens);
    metrics.emplace_back("mtp_standard_tokens", static_cast<double>(stats.standard_tokens));
    metrics.emplace_back("mtp_park_count", static_cast<double>(stats.park_count));
    metrics.emplace_back("mtp_reentry_probes", static_cast<double>(stats.reentry_probes));
    metrics.emplace_back(
        "mtp_selected_depth", static_cast<double>(stats.selected_depth));
    for (std::size_t depth = 0; depth < stats.depth_cycles.size(); ++depth) {
        metrics.emplace_back(
            "mtp_depth_" + std::to_string(depth) + "_cycles",
            static_cast<double>(stats.depth_cycles[depth]));
    }
    for (std::size_t position = 0;
            position < stats.position_drafted.size(); ++position) {
        metrics.emplace_back(
            "mtp_position_" + std::to_string(position + 1) +
                "_acceptance_rate",
            stats.position_drafted[position] == 0
                ? 0.0
                : static_cast<double>(stats.position_accepted[position]) /
                    stats.position_drafted[position]);
    }
    for (std::size_t depth = 0;
            depth < stats.measured_depth_ms.size(); ++depth) {
        metrics.emplace_back(
            "mtp_depth_" + std::to_string(depth) + "_cycle_ms",
            stats.measured_depth_ms[depth]);
    }
}

} // namespace mfq::engine::mtp
