#pragma once

#include "core.hpp"
#include "graph.hpp"
#include <algorithm>
#include <numeric>
#include <random>
#include <vector>
#include <omp.h>

namespace evtq {

template <typename DistanceFn, typename ErrorFn>
std::vector<SearchResult> select_neighbors(const std::vector<SearchResult>& candidates, size_t R, DistanceFn distance_fn, ErrorFn error_fn) {
    if (candidates.size() <= R) return {candidates.begin(), candidates.end()};

    std::vector<SearchResult> selected;
    selected.reserve(R);

    for (size_t i = 0; i < candidates.size() && selected.size() < R; ++i) {
        bool should_add = true;
        float err_c = error_fn(candidates[i].id);
        for (const auto& existing : selected) {
            if (distance_fn(candidates[i].id, existing.id) < candidates[i].distance + err_c + error_fn(existing.id)) { should_add = false; break; }
        }
        if (should_add) selected.push_back(candidates[i]);
    }

    if (selected.size() < R) {
        for (size_t i = 0; i < candidates.size() && selected.size() < R; ++i) {
            bool already = false;
            for (const auto& s : selected) if (s.id == candidates[i].id) { already = true; break; }
            if (!already) selected.push_back(candidates[i]);
        }
    }
    return selected;
}

namespace graph_refinement {

// RNN-Descent construction (arXiv 2310.20419, Algorithms 4-6)
// Parameters from paper Section 5.1: S=20, R=96, T1=4, T2=15

inline constexpr size_t RNN_INIT_DEGREE = 20;    // S: initial random graph degree
inline constexpr size_t RNN_MAX_DEGREE = 96;      // R: max degree during construction
inline constexpr size_t RNN_OUTER_ITERS = 4;      // T1: outer loop count
inline constexpr size_t RNN_INNER_ITERS = 15;     // T2: inner loop count per outer

struct NeighborEntry {
    NodeId id;
    float distance;
    bool is_new;
    bool operator<(const NeighborEntry& o) const { return distance < o.distance; }
};

struct ConstructionNode {
    std::vector<NeighborEntry> neighbors;
};

struct EdgeSwap { NodeId target, source; };

template <size_t D>
void init_random_graph(std::vector<ConstructionNode>& cg, const RaBitQGraph<D>& graph, size_t n) {
    #pragma omp parallel
    {
        std::mt19937 rng(static_cast<uint32_t>(n + omp_get_thread_num()));
        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            std::uniform_int_distribution<uint32_t> dist(0, static_cast<uint32_t>(n - 1));
            cg[i].neighbors.clear();
            cg[i].neighbors.reserve(RNN_MAX_DEGREE);
            for (size_t j = 0; j < RNN_INIT_DEGREE; ++j) {
                uint32_t v;
                do { v = dist(rng); } while (v == static_cast<uint32_t>(i));
                cg[i].neighbors.push_back({v, graph.distance_between(static_cast<NodeId>(i), v), true});
            }
        }
    }
}

template <size_t D>
void update_neighbors(std::vector<ConstructionNode>& cg, const RaBitQGraph<D>& graph, size_t n) {
    int num_threads = omp_get_max_threads();
    std::vector<std::vector<EdgeSwap>> thread_swaps(num_threads);

    // Phase 1: per-node RNG pruning with edge swap collection
    #pragma omp parallel
    {
        auto& local_swaps = thread_swaps[omp_get_thread_num()];
        local_swaps.clear();

        #pragma omp for schedule(dynamic, 64)
        for (size_t i = 0; i < n; ++i) {
            auto& node = cg[i];
            auto uid = static_cast<NodeId>(i);

            std::sort(node.neighbors.begin(), node.neighbors.end());

            std::vector<NeighborEntry> selected;
            selected.reserve(node.neighbors.size());

            for (auto& v_entry : node.neighbors) {
                bool keep = true;
                for (auto& w_entry : selected) {
                    if (!v_entry.is_new && !w_entry.is_new) continue;
                    float dist_vw = graph.distance_between(v_entry.id, w_entry.id);
                    if (v_entry.distance >= dist_vw) {
                        keep = false;
                        local_swaps.push_back({w_entry.id, v_entry.id});
                        break;
                    }
                }
                if (keep) selected.push_back(v_entry);
            }

            for (auto& e : selected) e.is_new = false;
            node.neighbors = std::move(selected);
        }
    }

    // Phase 2: apply deferred edge swaps grouped by target
    std::vector<std::vector<NodeId>> inbox(n);
    for (auto& swaps : thread_swaps)
        for (auto& [target, source] : swaps)
            inbox[target].push_back(source);

    std::vector<NodeId> targets;
    targets.reserve(n);
    for (size_t i = 0; i < n; ++i)
        if (!inbox[i].empty()) targets.push_back(static_cast<NodeId>(i));

    #pragma omp parallel for schedule(dynamic, 64)
    for (size_t idx = 0; idx < targets.size(); ++idx) {
        NodeId w = targets[idx];
        auto& node = cg[w];

        for (NodeId v : inbox[w]) {
            bool exists = false;
            for (auto& e : node.neighbors) {
                if (e.id == v) { exists = true; break; }
            }
            if (!exists) {
                node.neighbors.push_back({v, graph.distance_between(w, v), true});
            }
        }

        if (node.neighbors.size() > RNN_MAX_DEGREE) {
            std::partial_sort(node.neighbors.begin(), node.neighbors.begin() + RNN_MAX_DEGREE, node.neighbors.end());
            node.neighbors.resize(RNN_MAX_DEGREE);
        }
    }
}

template <size_t D>
void add_reverse_edges(std::vector<ConstructionNode>& cg, const RaBitQGraph<D>& graph, size_t n) {
    // Collect all reverse edges
    std::vector<std::vector<NodeId>> reverse_inbox(n);
    for (size_t i = 0; i < n; ++i)
        for (auto& e : cg[i].neighbors)
            reverse_inbox[e.id].push_back(static_cast<NodeId>(i));

    // Add reverse edges, mark as new
    #pragma omp parallel for schedule(dynamic, 64)
    for (size_t i = 0; i < n; ++i) {
        auto& node = cg[i];
        auto uid = static_cast<NodeId>(i);

        for (NodeId src : reverse_inbox[i]) {
            bool exists = false;
            for (auto& e : node.neighbors) {
                if (e.id == src) { exists = true; break; }
            }
            if (!exists) {
                node.neighbors.push_back({src, graph.distance_between(uid, src), true});
            }
        }

        // Trim out-degree to R (keep R shortest)
        if (node.neighbors.size() > RNN_MAX_DEGREE) {
            std::partial_sort(node.neighbors.begin(), node.neighbors.begin() + RNN_MAX_DEGREE, node.neighbors.end());
            node.neighbors.resize(RNN_MAX_DEGREE);
        }
    }
}

template <size_t D, typename EncType>
void finalize_graph(std::vector<ConstructionNode>& cg, RaBitQGraph<D>& graph, const EncType& encoder, size_t n) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        auto uid = static_cast<NodeId>(i);
        auto& node = cg[i];

        // Step A: SNG diversity trim to GRAPH_DEGREE
        std::vector<SearchResult> candidates;
        candidates.reserve(node.neighbors.size());
        for (auto& e : node.neighbors) candidates.push_back({e.id, e.distance});
        std::sort(candidates.begin(), candidates.end());

        auto dist_fn = [&](NodeId a, NodeId b) -> float { return graph.distance_between(a, b); };
        auto error_fn = [&](NodeId nid) -> float { return encoder.inv_sqrt_d_ * graph.get_code(nid).centered_norm; };
        auto selected = select_neighbors(candidates, GRAPH_DEGREE, dist_fn, error_fn);

        // Step B: quantize and write to NbitFastScanNeighborBlock
        auto& nb = graph.get_neighbors(uid);
        nb.count = 0;
        const float* vec_node = graph.get_vector(uid);
        alignas(SIMD_ALIGNMENT) float rotated_parent[D];
        encoder.rotate_and_normalize(vec_node, rotated_parent);

        for (size_t j = 0; j < selected.size(); ++j) {
            auto [code, aux] = encoder.compute_neighbor_aux_nbit(vec_node, graph.get_vector(selected[j].id), rotated_parent);
            nb.set_neighbor(j, selected[j].id, code, aux);
        }
    }
}

template <size_t D, typename EncType>
void rnn_descent_build(RaBitQGraph<D>& graph, const EncType& encoder) {
    size_t n = graph.size();

    std::vector<ConstructionNode> cg(n);
    init_random_graph<D>(cg, graph, n);

    for (size_t t1 = 0; t1 < RNN_OUTER_ITERS; ++t1) {
        for (size_t t2 = 0; t2 < RNN_INNER_ITERS; ++t2)
            update_neighbors<D>(cg, graph, n);
        if (t1 + 1 < RNN_OUTER_ITERS)
            add_reverse_edges<D>(cg, graph, n);
    }

    finalize_graph<D>(cg, graph, encoder, n);

    const auto& centroid_vec = encoder.get_centroid();
    alignas(SIMD_ALIGNMENT) float centroid[D]{};
    std::copy_n(centroid_vec.data(), centroid_vec.size(), centroid);
    graph.set_entry_point(graph.find_hub_entry(centroid));
}

}
}
