#pragma once

#include "core.hpp"
#include "graph.hpp"
#include "search.hpp"
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstdint>
#include <random>

#include <omp.h>

namespace evtq {

template <typename DistanceFn, typename ErrorFn>
std::vector<SearchResult> select_neighbors(const std::vector<SearchResult>& candidates, size_t R, DistanceFn distance_fn, ErrorFn error_fn) {
    if (candidates.size() <= R) {
        return {candidates.begin(), candidates.end()};
    }

    std::vector<SearchResult> selected;
    selected.reserve(R);

    for (size_t i = 0; i < candidates.size() && selected.size() < R; ++i) {
        bool should_add = true;
        float err_candidate = error_fn(candidates[i].id);
        float dist_cq = candidates[i].distance;

        for (const auto& existing : selected) {
            float dist_ce = distance_fn(candidates[i].id, existing.id);
            float margin = err_candidate + error_fn(existing.id);
            if (dist_ce < dist_cq + margin) {
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
            bool already = false;
            for (const auto& s : selected) {
                if (s.id == candidates[i].id) { already = true; break; }
            }
            if (!already) {
                selected.push_back(candidates[i]);
            }
        }
    }

    return selected;
}

namespace graph_refinement {

template <size_t D, typename EncType>
void prune_and_write(RaBitQGraph<D>& graph, const EncType& encoder, NodeId node, std::vector<SearchResult>& candidates) {
    auto dist_fn = [&](NodeId a, NodeId b) -> float {
        return graph.distance_between(a, b);
    };
    auto error_fn = [&](NodeId nid) -> float {
        return encoder.inv_sqrt_d_ * graph.get_code(nid).centered_norm;
    };

    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) {
                  return a.id < b.id || (a.id == b.id && a.distance < b.distance);
              });
    candidates.erase(
        std::unique(candidates.begin(), candidates.end(),
                    [](const auto& a, const auto& b) { return a.id == b.id; }),
        candidates.end());
    std::sort(candidates.begin(), candidates.end());

    auto selected = select_neighbors(candidates, GRAPH_DEGREE, dist_fn, error_fn);

    auto& nb = graph.get_neighbors(node);
    nb.count = 0;
    const float* vec_node = graph.get_vector(node);

    alignas(SIMD_ALIGNMENT) float rotated_parent[D];
    encoder.rotate_and_normalize(vec_node, rotated_parent);

    for (size_t j = 0; j < selected.size(); ++j) {
        auto [code, aux] = encoder.compute_neighbor_aux_nbit(
            vec_node, graph.get_vector(selected[j].id), rotated_parent);
        nb.set_neighbor(j, selected[j].id, code, aux);
    }
}

template <size_t D, typename EncType>
void vamana_build(RaBitQGraph<D>& graph, const EncType& encoder) {
    size_t n = graph.size();

    std::mt19937 rng(static_cast<uint32_t>(n));

    for (size_t i = 0; i < n; ++i) {
        auto& nb = graph.get_neighbors(static_cast<NodeId>(i));
        nb.count = 0;
        std::uniform_int_distribution<uint32_t> dist(0, static_cast<uint32_t>(n - 1));
        for (size_t j = 0; j < GRAPH_DEGREE; ++j) {
            uint32_t v;
            do { v = dist(rng); } while (v == static_cast<uint32_t>(i));
            nb.neighbor_ids[j] = v;
        }
        nb.count = GRAPH_DEGREE;
    }

    const auto& centroid_vec = encoder.get_centroid();
    alignas(SIMD_ALIGNMENT) float centroid[D]{};
    std::copy_n(centroid_vec.data(), centroid_vec.size(), centroid);
    NodeId medoid = graph.find_hub_entry(centroid);
    graph.set_entry_point(medoid);

    std::vector<NodeId> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);

    VisitationTable visited(n);

    for (;;) {
        std::shuffle(perm.begin(), perm.end(), rng);
        size_t updates = 0;

        for (NodeId uid : perm) {
            const float* vec = graph.get_vector(uid);

            auto candidates = rabitq_search::search_graph_ceiling<D>(
                vec, graph, GRAPH_DEGREE, visited, medoid);

            const auto& nb = graph.get_neighbors(uid);
            for (size_t j = 0; j < nb.size(); ++j) {
                NodeId nid = nb.neighbor_ids[j];
                if (nid != INVALID_NODE)
                    candidates.push_back({nid, graph.distance_between(uid, nid)});
            }

            std::erase_if(candidates, [uid](const SearchResult& r) { return r.id == uid; });

            const auto& old_nb = graph.get_neighbors(uid);
            uint32_t old_first = (old_nb.size() > 0) ? old_nb.neighbor_ids[0] : INVALID_NODE;

            prune_and_write<D>(graph, encoder, uid, candidates);

            const auto& updated_nb = graph.get_neighbors(uid);
            if (updated_nb.neighbor_ids[0] != old_first) ++updates;

            for (size_t j = 0; j < updated_nb.size(); ++j) {
                NodeId v = updated_nb.neighbor_ids[j];
                if (v == INVALID_NODE) continue;

                const auto& vnb = graph.get_neighbors(v);
                std::vector<SearchResult> v_cands;
                v_cands.reserve(vnb.size() + 1);
                for (size_t k = 0; k < vnb.size(); ++k) {
                    NodeId w = vnb.neighbor_ids[k];
                    if (w != INVALID_NODE)
                        v_cands.push_back({w, graph.distance_between(v, w)});
                }
                v_cands.push_back({uid, graph.distance_between(v, uid)});

                prune_and_write<D>(graph, encoder, v, v_cands);
            }
        }

        if (updates == 0) break;
    }

    NodeId hub = graph.find_hub_entry(centroid);
    graph.set_entry_point(hub);
}

}
}
