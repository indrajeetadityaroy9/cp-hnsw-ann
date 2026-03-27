#pragma once

#include "../core/core.hpp"
#include "rabitq_graph.hpp"
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstdint>
#include <random>
#include <atomic>

#include <omp.h>

namespace cphnsw {

// ============================================================
// Neighbor Selection (formerly neighbor_selection.hpp)
// ============================================================

template <typename DistanceFn, typename ErrorFn>
std::vector<SearchResult> select_neighbors_alpha_cng(
    std::vector<SearchResult> candidates,
    size_t R,
    DistanceFn distance_fn,
    ErrorFn error_fn,
    float alpha,
    float tau,
    float alpha_max = 0.0f)
{
    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) {
                  return a.id < b.id || (a.id == b.id && a.distance < b.distance);
              });
    candidates.erase(
        std::unique(candidates.begin(), candidates.end(),
                    [](const auto& a, const auto& b) { return a.id == b.id; }),
        candidates.end());

    std::sort(candidates.begin(), candidates.end());

    if (candidates.size() <= R) return candidates;

    float local_alpha = std::max(alpha, 1.0f);

    std::vector<SearchResult> selected;
    selected.reserve(R);

    for (size_t i = 0; i < candidates.size() && selected.size() < R; ++i) {
        bool should_add = true;
        float err_candidate = error_fn(candidates[i].id);
        float dist_cq = candidates[i].distance;

        for (const auto& existing : selected) {
            float dist_ce = distance_fn(candidates[i].id, existing.id);
            float err_existing = error_fn(existing.id);
            float margin = err_candidate + err_existing;

            float threshold = local_alpha * dist_cq + margin - (local_alpha - 1.0f) * tau;
            if (dist_ce < threshold) {
                should_add = false;
                break;
            }
        }
        if (should_add) {
            selected.push_back(candidates[i]);
        }
    }

    if (selected.size() < R) {
        for (size_t i = 0; i < candidates.size() && selected.size() < R; ++i) {
            bool already_selected = false;
            for (const auto& s : selected) {
                if (s.id == candidates[i].id) { already_selected = true; break; }
            }
            if (!already_selected) {
                selected.push_back(candidates[i]);
            }
        }
    }

    return selected;
}

// ============================================================
// Graph Refinement
// ============================================================

constexpr size_t isqrt(size_t n) {
    if (n < 2) return n;
    size_t x = n, y = (x + 1) / 2;
    while (y < x) { x = y; y = (x + n / x) / 2; }
    return x;
}

struct GraphStats {
    float alpha;
    float tau;
    float alpha_max;
};

inline float stddev(const float* data, size_t n) {
    float mean = 0.0f;
    for (size_t i = 0; i < n; ++i) mean += data[i];
    mean /= static_cast<float>(n);
    float var = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float d = data[i] - mean;
        var += d * d;
    }
    return std::sqrt(var / static_cast<float>(n));
}
namespace graph_refinement {

template <size_t D, size_t BitWidth, typename EncType>
void prune_and_write(RaBitQGraph<D, BitWidth>& graph, const EncType& encoder,
                     NodeId node, std::vector<SearchResult>& candidates,
                     float alpha, float tau, float error_tolerance = 0.0f,
                     float alpha_max = 0.0f) {
    auto dist_fn = [&](NodeId a, NodeId b) -> float {
        return l2_distance_simd<D>(graph.get_vector(a), graph.get_vector(b));
    };
    auto error_fn = [&](NodeId nid) -> float {
        const auto& code = graph.get_code(nid);
        return error_tolerance * code.nop;
    };

    auto selected = select_neighbors_alpha_cng(
        std::move(candidates), GRAPH_DEGREE, dist_fn, error_fn, alpha, tau, alpha_max);

    auto& nb = graph.get_neighbors(node);
    nb.count = 0;
    const float* vec_node = graph.get_vector(node);

    alignas(32) float rotated_parent[D];
    encoder.rotate_raw_vector(vec_node, rotated_parent);

    for (size_t j = 0; j < selected.size(); ++j) {
        NodeId v = selected[j].id;
        const float* vec_v = graph.get_vector(v);

        auto [code, aux] = encoder.compute_neighbor_aux_nbit(
            vec_node, vec_v, rotated_parent);
        nb.set_neighbor(j, v, code, aux);
    }
}


template <size_t D, size_t BitWidth>
void init_working_random(
    const RaBitQGraph<D, BitWidth>& graph,
    std::vector<std::vector<SearchResult>>& working,
    size_t actual_threads)
{
    size_t n = graph.size();

    #pragma omp parallel num_threads(actual_threads)
    {
        int tid = omp_get_thread_num();
        std::mt19937 rng(static_cast<uint32_t>(42 + static_cast<uint64_t>(tid)));

        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            NodeId u = static_cast<NodeId>(i);
            const float* vec_u = graph.get_vector(u);
            working[i].clear();

            std::uniform_int_distribution<size_t> dist(0, n - 1);

            std::vector<SearchResult> candidates;
            size_t pool_size = std::min(
                static_cast<size_t>(static_cast<double>(GRAPH_DEGREE) *
                    std::ceil(std::log(static_cast<double>(n) / static_cast<double>(GRAPH_DEGREE)))),
                n - 1);
            candidates.reserve(pool_size);

            // Bitset for O(1) duplicate detection instead of O(pool_size) scan
            std::vector<bool> seen(n, false);
            seen[u] = true;
            while (candidates.size() < pool_size) {
                NodeId v = static_cast<NodeId>(dist(rng));
                if (seen[v]) continue;
                seen[v] = true;
                float d = l2_distance_simd<D>(vec_u, graph.get_vector(v));
                candidates.push_back({v, d});
            }
            std::sort(candidates.begin(), candidates.end(),
                      [](const auto& a, const auto& b) { return a.distance < b.distance; });
            size_t keep = std::min(candidates.size(), static_cast<size_t>(GRAPH_DEGREE));
            working[i].assign(candidates.begin(), candidates.begin() + keep);
        }
    }
}


template <size_t D, size_t BitWidth>
size_t nndescent_join_pass(
    const RaBitQGraph<D, BitWidth>& graph,
    std::vector<std::vector<SearchResult>>& working,
    std::vector<std::vector<uint8_t>>& new_flags,
    size_t actual_threads)
{
    size_t n = graph.size();

    struct SnapshotEntry {
        NodeId ids[GRAPH_DEGREE];
        uint8_t is_new[GRAPH_DEGREE];
        uint8_t count;
    };
    std::vector<SnapshotEntry> snapshot(n);

    for (size_t i = 0; i < n; ++i) {
        size_t sz = std::min(working[i].size(), static_cast<size_t>(GRAPH_DEGREE));
        snapshot[i].count = static_cast<uint8_t>(sz);
        for (size_t j = 0; j < sz; ++j) {
            snapshot[i].ids[j] = working[i][j].id;
            snapshot[i].is_new[j] = new_flags[i][j];
        }
        std::fill(new_flags[i].begin(), new_flags[i].end(), 0);
    }

    std::vector<std::vector<NodeId>> reverse(n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < snapshot[i].count; ++j) {
            NodeId v = snapshot[i].ids[j];
            if (v != INVALID_NODE) {
                reverse[v].push_back(static_cast<NodeId>(i));
            }
        }
    }

    std::atomic<size_t> total_updates{0};

    #pragma omp parallel num_threads(actual_threads)
    {
        std::vector<NodeId> candidates;
        candidates.reserve(GRAPH_DEGREE * GRAPH_DEGREE);

        #pragma omp for schedule(guided)
        for (size_t i = 0; i < n; ++i) {
            NodeId u = static_cast<NodeId>(i);
            const float* vec_u = graph.get_vector(u);
            auto& wl = working[i];

            bool has_new_forward = false;
            for (size_t j = 0; j < snapshot[i].count; ++j) {
                if (snapshot[i].is_new[j]) { has_new_forward = true; break; }
            }

            bool has_new_reverse = false;
            for (NodeId rv : reverse[i]) {
                for (size_t j = 0; j < snapshot[rv].count; ++j) {
                    if (snapshot[rv].is_new[j]) { has_new_reverse = true; break; }
                }
                if (has_new_reverse) break;
            }

            if (!has_new_forward && !has_new_reverse) continue;

            candidates.clear();

            // Bitset for O(1) membership test instead of O(R) scan per candidate
            thread_local std::vector<bool> wl_member;
            wl_member.assign(n, false);
            wl_member[u] = true;
            for (const auto& nb : wl) {
                if (nb.id != INVALID_NODE) wl_member[nb.id] = true;
            }
            auto is_current_or_self = [&](NodeId w) -> bool {
                return wl_member[w];
            };

            if (has_new_forward) {
                for (size_t j = 0; j < snapshot[i].count; ++j) {
                    if (!snapshot[i].is_new[j]) continue;
                    NodeId v = snapshot[i].ids[j];
                    if (v == INVALID_NODE) continue;
                    for (size_t k = 0; k < snapshot[v].count; ++k) {
                        NodeId w = snapshot[v].ids[k];
                        if (w != INVALID_NODE && !is_current_or_self(w)) {
                            candidates.push_back(w);
                        }
                    }
                }
            }

            for (NodeId rv : reverse[i]) {
                bool rv_has_new = false;
                for (size_t j = 0; j < snapshot[rv].count; ++j) {
                    if (snapshot[rv].is_new[j]) { rv_has_new = true; break; }
                }
                if (!rv_has_new) continue;

                for (size_t k = 0; k < snapshot[rv].count; ++k) {
                    NodeId w = snapshot[rv].ids[k];
                    if (w != INVALID_NODE && !is_current_or_self(w)) {
                        candidates.push_back(w);
                    }
                }
            }

            if (candidates.empty()) continue;

            std::sort(candidates.begin(), candidates.end());
            candidates.erase(
                std::unique(candidates.begin(), candidates.end()),
                candidates.end());

            size_t local_updates = 0;
            float worst = (wl.size() >= GRAPH_DEGREE)
                ? wl.back().distance
                : std::numeric_limits<float>::max();

            for (NodeId w : candidates) {
                float d = l2_distance_simd<D>(vec_u, graph.get_vector(w));
                if (d >= worst && wl.size() >= GRAPH_DEGREE) continue;

                if (wl.size() < GRAPH_DEGREE) {
                    wl.push_back({w, d});
                    new_flags[i].push_back(1);
                } else {
                    wl.back() = {w, d};
                    new_flags[i].back() = 1;
                }
                for (size_t p = wl.size() - 1; p > 0 && wl[p].distance < wl[p-1].distance; --p) {
                    std::swap(wl[p], wl[p-1]);
                    std::swap(new_flags[i][p], new_flags[i][p-1]);
                }

                worst = (wl.size() >= GRAPH_DEGREE) ? wl.back().distance : std::numeric_limits<float>::max();
                local_updates++;
            }

            if (local_updates > 0) {
                total_updates.fetch_add(local_updates, std::memory_order_relaxed);
            }
        }
    }

    return total_updates.load();
}


template <size_t D, size_t BitWidth>
GraphStats derive_graph_stats(
    const RaBitQGraph<D, BitWidth>& graph,
    const std::vector<std::vector<SearchResult>>& working,
    size_t sample_size)
{
    GraphStats stats;
    size_t n = working.size();
    size_t actual_sample = std::min(sample_size, n);
    std::mt19937 rng(static_cast<uint32_t>(42 + 1));
    std::vector<size_t> sample_indices(n);
    std::iota(sample_indices.begin(), sample_indices.end(), 0);
    std::shuffle(sample_indices.begin(), sample_indices.end(), rng);
    sample_indices.resize(actual_sample);

    std::vector<float> neighbor_dists;
    std::vector<float> inter_neighbor_dists;
    std::vector<float> nn_dists;
    neighbor_dists.reserve(actual_sample * GRAPH_DEGREE);
    inter_neighbor_dists.reserve(actual_sample * GRAPH_DEGREE);
    nn_dists.reserve(actual_sample);

    constexpr size_t inter_limit_val = std::min(2 * isqrt(GRAPH_DEGREE), GRAPH_DEGREE);

    for (size_t idx : sample_indices) {
        const auto& wl = working[idx];
        for (const auto& nb : wl) {
            if (nb.id != INVALID_NODE) {
                neighbor_dists.push_back(nb.distance);
            }
        }
        if (!wl.empty() && wl[0].id != INVALID_NODE) {
            nn_dists.push_back(wl[0].distance);
        }
        size_t inter_limit = std::min(wl.size(), inter_limit_val);
        for (size_t j = 0; j < inter_limit; ++j) {
            for (size_t k = j + 1; k < inter_limit; ++k) {
                if (wl[j].id == INVALID_NODE || wl[k].id == INVALID_NODE) continue;
                float d = l2_distance_simd<D>(
                    graph.get_vector(wl[j].id), graph.get_vector(wl[k].id));
                inter_neighbor_dists.push_back(d);
            }
        }
    }

    size_t nd_n = neighbor_dists.size();
    std::nth_element(neighbor_dists.begin(), neighbor_dists.begin() + nd_n / 4, neighbor_dists.end());
    float nd_q1 = neighbor_dists[nd_n / 4];
    std::nth_element(neighbor_dists.begin() + nd_n / 4, neighbor_dists.begin() + nd_n / 2, neighbor_dists.end());
    float neighbor_dist_median = neighbor_dists[nd_n / 2];
    std::nth_element(neighbor_dists.begin() + nd_n / 2, neighbor_dists.begin() + 3 * nd_n / 4, neighbor_dists.end());
    float nd_q3 = neighbor_dists[3 * nd_n / 4];
    float neighbor_q3_over_q1 = (nd_q1 > std::numeric_limits<float>::epsilon())
        ? nd_q3 / nd_q1 : 0.0f;

    float nn_dist_sigma = stddev(nn_dists.data(), nn_dists.size());

    size_t inter_q1_idx = inter_neighbor_dists.size() / 4;
    std::nth_element(inter_neighbor_dists.begin(), inter_neighbor_dists.begin() + inter_q1_idx, inter_neighbor_dists.end());
    float d_inter = inter_neighbor_dists[inter_q1_idx];

    if (d_inter < std::numeric_limits<float>::epsilon()) {
        stats.alpha = 1.0f;
    } else {
        stats.alpha = neighbor_dist_median / d_inter;
    }

    stats.alpha_max = neighbor_q3_over_q1;
    stats.alpha = std::clamp(stats.alpha, 1.0f, stats.alpha_max);

    stats.tau = nn_dist_sigma;

    return stats;
}


template <size_t D, size_t BitWidth, typename EncType>
void run_reverse_edge_pass(RaBitQGraph<D, BitWidth>& graph, const EncType& encoder,
                           float alpha, float tau, float error_tolerance,
                           size_t actual_threads,
                           float alpha_max = 0.0f) {
    size_t n = graph.size();

    std::vector<std::vector<SearchResult>> reverse_cands(n);
    for (size_t u = 0; u < n; ++u) {
        const auto& nb = graph.get_neighbors(static_cast<NodeId>(u));
        const float* vec_u = graph.get_vector(static_cast<NodeId>(u));
        for (size_t j = 0; j < nb.count; ++j) {
            NodeId v = nb.neighbor_ids[j];
            if (v == INVALID_NODE) continue;
            float d = l2_distance_simd<D>(vec_u, graph.get_vector(v));
            reverse_cands[v].push_back({static_cast<NodeId>(u), d});
        }
    }

    #pragma omp parallel for schedule(guided) num_threads(actual_threads)
    for (size_t i = 0; i < n; ++i) {
        NodeId v = static_cast<NodeId>(i);
        if (reverse_cands[v].empty()) continue;

        const auto& nb = graph.get_neighbors(v);
        std::vector<SearchResult> all;
        all.reserve(nb.count + reverse_cands[v].size());

        const float* vec_v = graph.get_vector(v);
        for (size_t j = 0; j < nb.count; ++j) {
            NodeId w = nb.neighbor_ids[j];
            if (w == INVALID_NODE) continue;
            float d = l2_distance_simd<D>(vec_v, graph.get_vector(w));
            all.push_back({w, d});
        }

        for (const auto& cand : reverse_cands[v]) {
            if (cand.id == v) continue;
            all.push_back(cand);
        }

        prune_and_write<D, BitWidth>(graph, encoder, v, all, alpha, tau, error_tolerance, alpha_max);
    }
}


template <size_t D, size_t BitWidth, typename EncType>
std::vector<NodeId>
optimize_graph_adaptive(RaBitQGraph<D, BitWidth>& graph, const EncType& encoder) {
    size_t n = graph.size();

    size_t actual_threads = static_cast<size_t>(omp_get_max_threads());
    float error_tolerance = 1.0f / std::sqrt(static_cast<float>(D));

    auto centroid = graph.compute_centroid();
    NodeId entry_point = graph.find_nearest_to_centroid(centroid);
    graph.set_entry_point(entry_point);

    std::vector<std::vector<SearchResult>> working(n);
    std::vector<std::vector<uint8_t>> new_flags(n);

    init_working_random<D, BitWidth>(graph, working, actual_threads);

    for (size_t i = 0; i < n; ++i) {
        new_flags[i].assign(working[i].size(), 1);
    }

    size_t total_edges = std::max(n * GRAPH_DEGREE, size_t(1));

    size_t updates_0 = nndescent_join_pass<D, BitWidth>(
        graph, working, new_flags, actual_threads);
    float rate_0 = static_cast<float>(updates_0) / static_cast<float>(total_edges);

    size_t updates_1 = nndescent_join_pass<D, BitWidth>(
        graph, working, new_flags, actual_threads);
    float rate_1 = static_cast<float>(updates_1) / static_cast<float>(total_edges);

    float decay_ratio = (rate_0 > std::numeric_limits<float>::epsilon()) ? rate_1 / rate_0 : 0.0f;
    // Natural domain of exponential smoothing factor: [0, 1]
    // α = 1-r is the optimal weight for a geometrically decaying process
    float ema_alpha = std::clamp(1.0f - decay_ratio, 0.0f, 1.0f);

    float converge_rate = 1.0f / static_cast<float>(total_edges);

    size_t min_rounds;
    if (decay_ratio > 0.0f && decay_ratio < 1.0f && rate_0 > converge_rate) {
        min_rounds = static_cast<size_t>(std::ceil(
            std::log(converge_rate / rate_0) / std::log(decay_ratio)));
    } else {
        min_rounds = 2;
    }

    float ema_rate = ema_alpha * rate_1 + (1.0f - ema_alpha) * rate_0;

    for (size_t round = 2; ; ++round) {
        size_t updates = nndescent_join_pass<D, BitWidth>(
            graph, working, new_flags, actual_threads);
        float rate = static_cast<float>(updates) / static_cast<float>(total_edges);

        ema_rate = ema_alpha * rate + (1.0f - ema_alpha) * ema_rate;

        if (round >= min_rounds && ema_rate < converge_rate) break;
    }

    size_t alpha_sample = static_cast<size_t>(std::sqrt(static_cast<double>(n)));
    GraphStats stats = derive_graph_stats<D, BitWidth>(graph, working, alpha_sample);

    float alpha = stats.alpha;
    float tau = stats.tau;
    float alpha_max = stats.alpha_max;

    #pragma omp parallel for schedule(guided) num_threads(actual_threads)
    for (size_t i = 0; i < n; ++i) {
        NodeId u = static_cast<NodeId>(i);
        std::vector<SearchResult> candidates;
        candidates.reserve(working[i].size());
        for (const auto& nb : working[i]) {
            if (nb.id != INVALID_NODE) {
                candidates.push_back({nb.id, nb.distance});
            }
        }
        prune_and_write<D, BitWidth>(graph, encoder, u, candidates,
            alpha, tau, error_tolerance, alpha_max);
    }

    working.clear();
    working.shrink_to_fit();
    new_flags.clear();
    new_flags.shrink_to_fit();

    run_reverse_edge_pass<D, BitWidth>(graph, encoder, alpha, tau,
        error_tolerance, actual_threads, alpha_max);

    NodeId hub = graph.find_hub_entry(centroid);
    graph.set_entry_point(hub);

    return graph.reorder_bfs(hub);
}

}
}
