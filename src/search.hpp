#pragma once

#include "core.hpp"
#include "estimator.hpp"
#include "graph.hpp"
#include <algorithm>
#include <memory>
#include <queue>
#include <vector>

namespace evtq {

struct TwoLevelVisitationTable {
    explicit TwoLevelVisitationTable(size_t capacity)
        : capacity_(capacity), current_epoch_(0)
        , estimated_(std::make_unique<uint64_t[]>(capacity))
        , visited_(std::make_unique<uint64_t[]>(capacity)) {}

    TwoLevelVisitationTable(const TwoLevelVisitationTable&) = delete;
    TwoLevelVisitationTable& operator=(const TwoLevelVisitationTable&) = delete;

    uint64_t new_query() const {
        return ++current_epoch_;
    }

    bool check_and_mark_estimated(NodeId node_id, uint64_t query_id) const {
        if (estimated_[node_id] == query_id) return true;
        estimated_[node_id] = query_id;
        return false;
    }

    void mark_visited(NodeId node_id, uint64_t query_id) const {
        visited_[node_id] = query_id;
    }

    bool is_visited(NodeId node_id, uint64_t query_id) const {
        return visited_[node_id] == query_id;
    }

    void prefetch_estimated(NodeId node_id) const {
        prefetch_t<0>(reinterpret_cast<const char*>(&estimated_[node_id]));
    }

    void resize(size_t new_capacity) {
        estimated_ = std::make_unique<uint64_t[]>(new_capacity);
        visited_ = std::make_unique<uint64_t[]>(new_capacity);
        capacity_ = new_capacity;
    }

    size_t capacity() const { return capacity_; }

    mutable std::unique_ptr<uint64_t[]> estimated_;
    mutable std::unique_ptr<uint64_t[]> visited_;
    size_t capacity_;
    mutable uint64_t current_epoch_;
};

template <typename T>
struct BoundedMaxHeap {
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
        return data_.front().distance;
    }

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

template <size_t D>
std::vector<SearchResult> search(RaBitQQuery<D> query, const float* raw_query, const RaBitQGraph<D>& graph, size_t k, TwoLevelVisitationTable& visited, NodeId entry, const QRCTCalibration& cal) {
    uint64_t query_id = visited.new_query();

    std::priority_queue<BeamEntry, std::vector<BeamEntry>, std::greater<BeamEntry>> beam;
    BoundedMaxHeap<SearchResult> nn(k);

    float query_norm_sq = dot_product_simd<D>(raw_query, raw_query);

    auto exact_l2 = [&](NodeId id) -> float {
        return graph.query_distance(raw_query, query_norm_sq, id);
    };

    float entry_est = exact_l2(entry);
    beam.push({entry_est, 0.0f, entry});
    visited.check_and_mark_estimated(entry, query_id);

    estimator::NeighborEstimates<D> estimates;

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

        if (nn.size() >= k && current.lower_bound > nn.worst_distance()) continue;

        if (!beam.empty()) {
            graph.prefetch_vertex(beam.top().id);
            graph.prefetch_vector(beam.top().id);
            graph.prefetch_norm(beam.top().id);
        }

        visited.mark_visited(current.id, query_id);

        float exact_dist = exact_l2(current.id);
        nn.push({current.id, exact_dist});

        if (nn.size() >= k) {
            float R_B = 0.0f;
            float d_k_sq = nn.worst_distance();
            thread_local std::vector<BeamEntry> risk_buf;
            risk_buf.clear();

            while (!beam.empty()) {
                BeamEntry e = beam.top();
                beam.pop();
                if (visited.is_visited(e.id, query_id)) continue;
                float surv = qrct::gpd_survival(e.est_distance / d_k_sq, cal);
                if (surv == 0.0f) break;
                risk_buf.push_back(e);
                R_B += surv;
            }
            for (const auto& e : risk_buf) beam.push(e);

            if (R_B <= cal.delta) return nn.extract_sorted();
        }

        const auto& nb = graph.get_neighbors(current.id);
        size_t n_neighbors = nb.size();
        float dist_qp_sq = exact_dist;
        bool warmup = (nn.size() < k);
        estimator::estimate_neighbors_for_search<D>(query, nb, dist_qp_sq, nn.worst_distance(), !warmup, estimates);

        constexpr size_t VISIT_PREFETCH = CACHE_LINE_SIZE / sizeof(uint64_t);
        size_t prefetch_count = std::min(n_neighbors, VISIT_PREFETCH);
        for (size_t i = 0; i < prefetch_count; ++i) {
            NodeId nid = nb.neighbor_ids[i];
            visited.prefetch_estimated(nid);
            graph.prefetch_norm(nid);
        }

        constexpr size_t PREFETCH_LOOKAHEAD = (RaBitQGraph<D>::PREFETCH_LINES + VISIT_PREFETCH - 1) / VISIT_PREFETCH;
        for (size_t i = 0; i < n_neighbors; ++i) {
            if (i + PREFETCH_LOOKAHEAD < n_neighbors) {
                NodeId future_id = nb.neighbor_ids[i + PREFETCH_LOOKAHEAD];
                graph.prefetch_vertex(future_id);
                graph.prefetch_vector(future_id);
                graph.prefetch_norm(future_id);
            }

            NodeId neighbor_id = nb.neighbor_ids[i];
            if (visited.check_and_mark_estimated(neighbor_id, query_id)) continue;

            if (warmup) {
                float exact = exact_l2(neighbor_id);
                nn.push({neighbor_id, exact});
                beam.push({exact, exact, neighbor_id});
                graph.prefetch_vertex(neighbor_id);
                continue;
            }

            float est_dist = estimates.est_distances[i];
            float lower = estimates.lower_bounds[i];
            if (nn.size() >= k && lower >= nn.worst_distance()) continue;

            if (est_dist < nn.worst_distance()) {
                float exact = exact_l2(neighbor_id);
                nn.push({neighbor_id, exact});
                beam.push({exact, lower, neighbor_id});
            } else {
                beam.push({est_dist, lower, neighbor_id});
            }
            graph.prefetch_vertex(neighbor_id);
        }
    }

    return nn.extract_sorted();
}

}
}
