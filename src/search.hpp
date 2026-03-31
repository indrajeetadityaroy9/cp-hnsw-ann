#pragma once

#include "core.hpp"
#include "fastscan.hpp"
#include "graph.hpp"
#include <memory>
#include <vector>
#include <queue>
#include <algorithm>
#include <limits>

namespace evtq {

// ============================================================
// Two-Level Visitation Table
// ============================================================

class TwoLevelVisitationTable {
public:
    explicit TwoLevelVisitationTable(size_t capacity)
        : capacity_(capacity), current_epoch_(0) {
        estimated_ = std::make_unique<uint64_t[]>(capacity);
        visited_ = std::make_unique<uint64_t[]>(capacity);
        std::memset(estimated_.get(), 0, capacity * sizeof(uint64_t));
        std::memset(visited_.get(), 0, capacity * sizeof(uint64_t));
    }

    TwoLevelVisitationTable(const TwoLevelVisitationTable&) = delete;
    TwoLevelVisitationTable& operator=(const TwoLevelVisitationTable&) = delete;
    TwoLevelVisitationTable(TwoLevelVisitationTable&&) = delete;
    TwoLevelVisitationTable& operator=(TwoLevelVisitationTable&&) = delete;

    uint64_t new_query() const {
        return ++current_epoch_;
    }

    bool check_and_mark_estimated(NodeId node_id, uint64_t query_id) const {
        if (estimated_[node_id] == query_id) return true;
        estimated_[node_id] = query_id;
        return false;
    }

    bool check_and_mark_visited(NodeId node_id, uint64_t query_id) const {
        if (visited_[node_id] == query_id) return true;
        visited_[node_id] = query_id;
        return false;
    }

    bool is_visited(NodeId node_id, uint64_t query_id) const {
        return visited_[node_id] == query_id;
    }

    void prefetch_estimated(NodeId node_id) const {
        prefetch_t<0>(reinterpret_cast<const char*>(&estimated_[node_id]));
    }

    void resize(size_t new_capacity) {
        auto new_est = std::make_unique<uint64_t[]>(new_capacity);
        auto new_vis = std::make_unique<uint64_t[]>(new_capacity);
        std::memcpy(new_est.get(), estimated_.get(), capacity_ * sizeof(uint64_t));
        std::memcpy(new_vis.get(), visited_.get(), capacity_ * sizeof(uint64_t));
        std::memset(new_est.get() + capacity_, 0, (new_capacity - capacity_) * sizeof(uint64_t));
        std::memset(new_vis.get() + capacity_, 0, (new_capacity - capacity_) * sizeof(uint64_t));
        estimated_ = std::move(new_est);
        visited_ = std::move(new_vis);
        capacity_ = new_capacity;
    }

    size_t capacity() const { return capacity_; }

private:
    mutable std::unique_ptr<uint64_t[]> estimated_;
    mutable std::unique_ptr<uint64_t[]> visited_;
    size_t capacity_;
    mutable uint64_t current_epoch_;
};

// ============================================================
// RaBitQ Search
// ============================================================

template <typename T>
class BoundedMaxHeap {
public:
    explicit BoundedMaxHeap(size_t capacity) : capacity_(capacity) {
        data_.reserve(capacity + 1);
    }

    size_t size() const { return data_.size(); }

    void push(const T& val) {
        if (data_.size() < capacity_) {
            data_.push_back(val);
            std::push_heap(data_.begin(), data_.end());
        } else if (val < data_.front()) {
            std::pop_heap(data_.begin(), data_.end());
            data_.back() = val;
            std::push_heap(data_.begin(), data_.end());
        }
    }

    std::vector<T> extract_sorted() {
        std::sort_heap(data_.begin(), data_.end());
        return std::move(data_);
    }

    float worst_distance() const {
        return data_.empty() ? std::numeric_limits<float>::max() : data_.front().distance;
    }

private:
    std::vector<T> data_;
    size_t capacity_;
};

namespace rabitq_search {

struct BeamEntry {
    float est_distance;
    float lower_bound;
    NodeId id;
    bool operator>(const BeamEntry& o) const { return est_distance > o.est_distance; }
};

template <size_t D, size_t BitWidth = 1>
std::vector<SearchResult> search(
    RaBitQQuery<D> query,
    const float* raw_query,
    const RaBitQGraph<D, BitWidth>& graph,
    size_t k,
    float gamma,
    TwoLevelVisitationTable& visited,
    NodeId entry)
{
    NodeId ep = entry;

    uint64_t query_id = visited.new_query();

    std::priority_queue<BeamEntry, std::vector<BeamEntry>,
                       std::greater<BeamEntry>> beam;
    BoundedMaxHeap<SearchResult> nn(k);

    float max_overest = 0.0f;
    float gamma_q = gamma;

    float query_norm_sq = dot_product_simd<D>(raw_query, raw_query);

    auto exact_l2 = [&](NodeId id) -> float {
        return query_norm_sq + graph.get_norm_sq(id)
               - 2.0f * dot_product_simd<D>(raw_query, graph.get_vector(id));
    };

    float ep_est = exact_l2(ep);
    beam.push({ep_est, 0.0f, ep});
    visited.check_and_mark_estimated(ep, query_id);

    alignas(SIMD_ALIGNMENT) uint32_t fastscan_sums[GRAPH_DEGREE];
    alignas(SIMD_ALIGNMENT) uint32_t msb_sums[GRAPH_DEGREE];
    alignas(SIMD_ALIGNMENT) float est_distances[GRAPH_DEGREE];
    alignas(SIMD_ALIGNMENT) float lower_bounds[GRAPH_DEGREE];

    while (!beam.empty()) {
        BeamEntry current;
        bool found = false;

        while (!beam.empty()) {
            current = beam.top();
            beam.pop();
            if (visited.is_visited(current.id, query_id)) continue;
            found = true;
            break;
        }
        if (!found) break;


        if (nn.size() >= k && current.est_distance > gamma_q * nn.worst_distance()) [[unlikely]] break;

        if (nn.size() >= k && current.lower_bound > nn.worst_distance()) continue;

        if (!beam.empty()) {
            graph.prefetch_vertex(beam.top().id);
            graph.prefetch_vector(beam.top().id);
            graph.prefetch_norm(beam.top().id);
        }

        visited.check_and_mark_visited(current.id, query_id);

        float exact_dist = exact_l2(current.id);
        nn.push({current.id, exact_dist});

        if (exact_dist != current.est_distance) {
            float overest = current.est_distance / exact_dist;
            max_overest = std::max(max_overest, overest);
            gamma_q = std::min(gamma, std::max(max_overest, 1.0f));
        }

        const auto& nb = graph.get_neighbors(current.id);
        size_t n_neighbors = nb.size();
        float dist_qp_sq = exact_dist;

        constexpr size_t BATCH = fastscan::BATCH_SIZE;
        size_t num_batches = (GRAPH_DEGREE + BATCH - 1) / BATCH;

        for (size_t batch = 0; batch < num_batches; ++batch) {
            size_t batch_start = batch * BATCH;
            if (batch_start >= n_neighbors) break;
            size_t batch_count = std::min(BATCH, n_neighbors - batch_start);

            if (batch + 1 < num_batches && (batch + 1) * BATCH < n_neighbors) {
                prefetch_t<0>(reinterpret_cast<const char*>(&nb.code_blocks[batch + 1]));
            }

            if constexpr (BitWidth >= 2) {
                // MSB shortcut path
                fastscan::compute_msb_only_inner_products<D, BitWidth>(
                    query.lut, nb.code_blocks[batch], msb_sums + batch_start);
                fastscan::convert_msb_to_lower_bounds<D, BitWidth>(
                    query, msb_sums + batch_start,
                    nb.centered_norm + batch_start, nb.code_ip + batch_start,
                    nb.code_parent_ip + batch_start, nb.popcounts + batch_start,
                    batch_count, lower_bounds + batch_start, dist_qp_sq);

                float threshold = nn.worst_distance();
                bool any_survivor = (nn.size() < k);
                if (!any_survivor) {
                    for (size_t j = 0; j < batch_count; ++j) {
                        if (lower_bounds[batch_start + j] < threshold) {
                            any_survivor = true;
                            break;
                        }
                    }
                }

                if (any_survivor) {
                    fastscan::compute_nbit_inner_products<D, BitWidth>(
                        query.lut, nb.code_blocks[batch],
                        fastscan_sums + batch_start, msb_sums + batch_start);
                    fastscan::convert_nbit_to_distances_with_bounds<D, BitWidth>(
                        query, fastscan_sums + batch_start,
                        msb_sums + batch_start,
                        nb.centered_norm + batch_start, nb.code_ip + batch_start,
                        nb.code_parent_ip + batch_start, nb.popcounts + batch_start,
                        nb.weighted_popcounts + batch_start,
                        batch_count, est_distances + batch_start,
                        lower_bounds + batch_start, dist_qp_sq);
                } else {
                    for (size_t j = 0; j < batch_count; ++j) {
                        est_distances[batch_start + j] = std::numeric_limits<float>::max();
                    }
                }
            } else {
                // BitWidth=1: compute directly, no MSB shortcut
                fastscan::compute_nbit_inner_products<D, BitWidth>(
                    query.lut, nb.code_blocks[batch],
                    fastscan_sums + batch_start, msb_sums + batch_start);
                fastscan::convert_nbit_to_distances_with_bounds<D, BitWidth>(
                    query, fastscan_sums + batch_start,
                    msb_sums + batch_start,
                    nb.centered_norm + batch_start, nb.code_ip + batch_start,
                    nb.code_parent_ip + batch_start, nb.popcounts + batch_start,
                    nb.weighted_popcounts + batch_start,
                    batch_count, est_distances + batch_start,
                    lower_bounds + batch_start, dist_qp_sq);
            }
        }


        bool warmup = (nn.size() < k);

        // One cache line of visitation table entries (uint64_t per entry)
        constexpr size_t VISIT_PREFETCH = CACHE_LINE_SIZE / sizeof(uint64_t);
        size_t prefetch_count = std::min(n_neighbors, VISIT_PREFETCH);
        for (size_t i = 0; i < prefetch_count; ++i) {
            NodeId nid = nb.neighbor_ids[i];
            visited.prefetch_estimated(nid);
            graph.prefetch_norm(nid);
        }

        // Lookahead: vertex prefetch depth divided by initial prefetch width.
        // Scales with vertex size relative to cache line capacity.
        constexpr size_t PREFETCH_LOOKAHEAD =
            (RaBitQGraph<D,BitWidth>::PREFETCH_LINES + VISIT_PREFETCH - 1) / VISIT_PREFETCH;
        for (size_t i = 0; i < n_neighbors; ++i) {
            if (i + PREFETCH_LOOKAHEAD < n_neighbors) {
                NodeId future_id = nb.neighbor_ids[i + PREFETCH_LOOKAHEAD];
                graph.prefetch_vertex(future_id);
                graph.prefetch_vector(future_id);
                graph.prefetch_norm(future_id);
            }

            NodeId neighbor_id = nb.neighbor_ids[i];
            if (visited.check_and_mark_estimated(neighbor_id, query_id)) continue;

            float dabs_threshold = (nn.size() >= k)
                ? gamma_q * nn.worst_distance()
                : std::numeric_limits<float>::max();

            if (warmup) {
                float exact = exact_l2(neighbor_id);
                nn.push({neighbor_id, exact});
                if (exact < dabs_threshold) {
                    beam.push({exact, exact, neighbor_id});
                }
                graph.prefetch_vertex(neighbor_id);
                continue;
            }

            float est_dist = est_distances[i];
            float lower = lower_bounds[i];
            if (nn.size() >= k && lower >= nn.worst_distance()) continue;

            if (est_dist < nn.worst_distance()) {
                float exact = exact_l2(neighbor_id);
                nn.push({neighbor_id, exact});
                if (exact < dabs_threshold) {
                    beam.push({exact, lower, neighbor_id});
                }

            } else if (est_dist < dabs_threshold) {
                beam.push({est_dist, lower, neighbor_id});
            }
            graph.prefetch_vertex(neighbor_id);
        }
    }

    return nn.extract_sorted();
}

}
}
