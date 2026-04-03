#pragma once

#include "core.hpp"
#include "estimator.hpp"
#include "graph.hpp"
#include <algorithm>
#include <cmath>
#include <memory>
#include <queue>
#include <vector>

namespace evtq {

struct VisitationTable {
    size_t capacity_;
    mutable std::unique_ptr<uint64_t[]> data_;
    mutable uint64_t current_epoch_;

    explicit VisitationTable(size_t capacity) : capacity_(capacity), data_(std::make_unique<uint64_t[]>(capacity * 2)), current_epoch_(0) {}
    VisitationTable(const VisitationTable&) = delete;
    VisitationTable& operator=(const VisitationTable&) = delete;

    uint64_t new_query() const { return ++current_epoch_; }
    bool check_and_mark_estimated(NodeId node_id, uint64_t query_id) const { if (data_[node_id] == query_id) return true; data_[node_id] = query_id; return false; }
    void mark_visited(NodeId node_id, uint64_t query_id) const { data_[capacity_ + node_id] = query_id; }
    bool is_visited(NodeId node_id, uint64_t query_id) const { return data_[capacity_ + node_id] == query_id; }
    void prefetch_estimated(NodeId node_id) const { prefetch_t<0>(reinterpret_cast<const char*>(&data_[node_id])); }
    void resize(size_t new_capacity) { data_ = std::make_unique<uint64_t[]>(new_capacity * 2); capacity_ = new_capacity; }
    size_t capacity() const { return capacity_; }
};

template <typename T>
struct BoundedMaxHeap {
    std::vector<T> data_;
    size_t capacity_;

    explicit BoundedMaxHeap(size_t capacity) : capacity_(capacity) { data_.reserve(capacity + 1); }
    size_t size() const { return data_.size(); }
    float worst_distance() const { return data_.front().distance; }

    void push(const T& val) {
        if (data_.size() < capacity_) { data_.push_back(val); std::push_heap(data_.begin(), data_.end()); }
        else if (val < data_.front()) { std::pop_heap(data_.begin(), data_.end()); data_.back() = val; std::push_heap(data_.begin(), data_.end()); }
    }

    std::vector<T> extract_sorted() { std::sort_heap(data_.begin(), data_.end()); return std::move(data_); }
};

namespace rabitq_search {

struct BeamEntry {
    float est_distance;
    NodeId id;
    bool operator>(const BeamEntry& o) const { return est_distance > o.est_distance; }
};

template <size_t D>
std::vector<SearchResult> search(const RaBitQQuery<D>& query, const float* raw_query, const RaBitQGraph<D>& graph, size_t k, VisitationTable& visited, NodeId entry, const GPDCalibration& cal) {
    uint64_t query_id = visited.new_query();
    std::priority_queue<BeamEntry, std::vector<BeamEntry>, std::greater<BeamEntry>> beam;
    BoundedMaxHeap<SearchResult> nn(k);

    float query_norm_sq = dot_product_simd<D>(raw_query, raw_query);
    auto exact_l2 = [&](NodeId id) -> float { return graph.query_distance(raw_query, query_norm_sq, id); };

    beam.push({exact_l2(entry), entry});
    visited.check_and_mark_estimated(entry, query_id);

    estimator::NeighborEstimates<D> estimates;
    float exhaustion_accumulator = 0.0f;

    while (!beam.empty()) {
        BeamEntry current;
        bool found = false;
        while (!beam.empty()) { current = beam.top(); beam.pop(); if (!visited.is_visited(current.id, query_id)) { found = true; break; } }
        if (!found) break;

        if (!beam.empty()) { graph.prefetch_vertex(beam.top().id); graph.prefetch_vector(beam.top().id); graph.prefetch_norm(beam.top().id); }

        visited.mark_visited(current.id, query_id);
        float worst_before = (nn.size() >= k) ? nn.worst_distance() : std::numeric_limits<float>::max();
        float exact_dist = exact_l2(current.id);
        nn.push({current.id, exact_dist});

        if (nn.size() >= k) {
            float d_k_sq = nn.worst_distance();
            if (exact_dist < worst_before) {
                exhaustion_accumulator = 0.0f;
            } else {
                float R_B = 0.0f;
                thread_local std::vector<BeamEntry> risk_buf;
                risk_buf.clear();
                while (!beam.empty()) {
                    BeamEntry e = beam.top(); beam.pop();
                    if (visited.is_visited(e.id, query_id)) continue;
                    float surv = rcgr::gpd_survival(e.est_distance / d_k_sq, cal);
                    risk_buf.push_back(e);
                    if (surv == 0.0f) break;
                    R_B += surv;
                }
                for (const auto& e : risk_buf) beam.push(e);
                exhaustion_accumulator += std::exp(-R_B);
                if (exhaustion_accumulator >= static_cast<float>(GRAPH_DEGREE)) return nn.extract_sorted();
            }
        }

        const auto& nb = graph.get_neighbors(current.id);
        size_t n_neighbors = nb.size();
        bool warmup = (nn.size() < k);
        estimator::estimate_neighbors<D>(query, nb, exact_dist, nn.worst_distance(), !warmup, estimates);

        constexpr size_t VISIT_PREFETCH = CACHE_LINE_SIZE / sizeof(uint64_t);
        for (size_t i = 0; i < std::min(n_neighbors, VISIT_PREFETCH); ++i) { visited.prefetch_estimated(nb.neighbor_ids[i]); graph.prefetch_norm(nb.neighbor_ids[i]); }

        constexpr size_t PREFETCH_LOOKAHEAD = (RaBitQGraph<D>::PREFETCH_LINES + VISIT_PREFETCH - 1) / VISIT_PREFETCH;
        for (size_t i = 0; i < n_neighbors; ++i) {
            if (i + PREFETCH_LOOKAHEAD < n_neighbors) {
                NodeId fid = nb.neighbor_ids[i + PREFETCH_LOOKAHEAD];
                graph.prefetch_vertex(fid); graph.prefetch_vector(fid); graph.prefetch_norm(fid);
            }
            NodeId nid = nb.neighbor_ids[i];
            if (visited.check_and_mark_estimated(nid, query_id)) continue;

            if (warmup) { beam.push({exact_l2(nid), nid}); graph.prefetch_vertex(nid); continue; }

            float est = estimates.est_distances[i], lower = estimates.lower_bounds[i];
            if (nn.size() >= k && lower >= nn.worst_distance()) continue;
            beam.push({(est < nn.worst_distance()) ? exact_l2(nid) : est, nid});
            graph.prefetch_vertex(nid);
        }
    }
    return nn.extract_sorted();
}

template <size_t D>
std::vector<SearchResult> search_graph_ceiling(const float* raw_query, const RaBitQGraph<D>& graph, size_t k, VisitationTable& visited, NodeId entry) {
    uint64_t query_id = visited.new_query();
    std::priority_queue<BeamEntry, std::vector<BeamEntry>, std::greater<BeamEntry>> beam;
    BoundedMaxHeap<SearchResult> nn(k);

    float query_norm_sq = dot_product_simd<D>(raw_query, raw_query);
    auto exact_l2 = [&](NodeId id) -> float { return graph.query_distance(raw_query, query_norm_sq, id); };

    beam.push({exact_l2(entry), entry});
    visited.check_and_mark_estimated(entry, query_id);

    while (!beam.empty()) {
        BeamEntry current;
        bool found = false;
        while (!beam.empty()) { current = beam.top(); beam.pop(); if (!visited.is_visited(current.id, query_id)) { found = true; break; } }
        if (!found) break;

        visited.mark_visited(current.id, query_id);
        nn.push({current.id, exact_l2(current.id)});

        const auto& nb = graph.get_neighbors(current.id);
        for (size_t i = 0; i < nb.size(); ++i) {
            NodeId nid = nb.neighbor_ids[i];
            if (!visited.check_and_mark_estimated(nid, query_id)) beam.push({exact_l2(nid), nid});
        }
    }
    return nn.extract_sorted();
}

}
}
