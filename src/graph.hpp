#pragma once

#include "core.hpp"
#include "fastscan.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <queue>
#include <span>
#include <vector>
#include <omp.h>

namespace evtq {

template <size_t D>
struct alignas(SIMD_ALIGNMENT) VertexSearchData {
    NbitRaBitQCode<D> code;
    NbitFastScanNeighborBlock<D> neighbors;
};

template <size_t D>
struct RaBitQGraph {
    using SearchDataType = VertexSearchData<D>;
    using RawVector = std::array<float, D>;

    size_t dim_;
    std::vector<SearchDataType, AlignedAllocator<SearchDataType>> search_data_;
    std::vector<RawVector> raw_vectors_;
    AlignedVector<float> norm_sq_;
    NodeId entry_point_ = INVALID_NODE;

    explicit RaBitQGraph(size_t dim) : dim_(dim) {}
    RaBitQGraph(const RaBitQGraph&) = delete;
    RaBitQGraph& operator=(const RaBitQGraph&) = delete;
    RaBitQGraph(RaBitQGraph&&) = default;
    RaBitQGraph& operator=(RaBitQGraph&&) = default;

    NodeId add_node(const NbitRaBitQCode<D>& code, std::span<const float> vec) {
        auto id = static_cast<NodeId>(search_data_.size());
        search_data_.emplace_back();
        search_data_.back().code = code;
        raw_vectors_.emplace_back();
        std::copy_n(vec.data(), dim_, raw_vectors_.back().data());
        std::fill_n(raw_vectors_.back().data() + dim_, D - dim_, 0.0f);
        const float* stored = raw_vectors_.back().data();
        norm_sq_.push_back(dot_product_simd<D>(stored, stored));
        if (entry_point_ == INVALID_NODE) entry_point_ = id;
        return id;
    }

    void reserve(size_t capacity) { search_data_.reserve(capacity); raw_vectors_.reserve(capacity); norm_sq_.reserve(capacity); }
    size_t size() const { return search_data_.size(); }
    NodeId entry_point() const { return entry_point_; }
    void set_entry_point(NodeId id) { entry_point_ = id; }

    const auto& get_search_data() const { return search_data_; }
    const auto& get_raw_vectors() const { return raw_vectors_; }
    const auto& get_norm_sq() const { return norm_sq_; }

    void restore_from_serialized(std::vector<SearchDataType, AlignedAllocator<SearchDataType>> sd, std::vector<RawVector> rv, AlignedVector<float> ns, NodeId ep) {
        search_data_ = std::move(sd); raw_vectors_ = std::move(rv); norm_sq_ = std::move(ns); entry_point_ = ep;
    }

    const NbitRaBitQCode<D>& get_code(NodeId id) const { return search_data_[id].code; }
    const float* get_vector(NodeId id) const { return raw_vectors_[id].data(); }
    float distance_between(NodeId a, NodeId b) const { return l2_distance_simd<D>(raw_vectors_[a].data(), raw_vectors_[b].data()); }
    float query_distance(const float* query, float query_norm_sq, NodeId id) const { return query_norm_sq + norm_sq_[id] - 2.0f * dot_product_simd<D>(query, raw_vectors_[id].data()); }
    void prefetch_norm(NodeId id) const { prefetch_t<1>(&norm_sq_[id]); }
    const NbitFastScanNeighborBlock<D>& get_neighbors(NodeId id) const { return search_data_[id].neighbors; }
    NbitFastScanNeighborBlock<D>& get_neighbors(NodeId id) { return search_data_[id].neighbors; }

    static constexpr size_t PREFETCH_LINES = std::min(sizeof(SearchDataType) / CACHE_LINE_SIZE, (sizeof(NbitRaBitQCode<D>) + sizeof(NbitFastScanCodeBlock<D>) + CACHE_LINE_SIZE - 1) / CACHE_LINE_SIZE);

    void prefetch_vertex(NodeId id) const {
        const char* base = reinterpret_cast<const char*>(&search_data_[id]);
        for (size_t line = 0; line < PREFETCH_LINES; ++line) prefetch_t<1>(base + line * CACHE_LINE_SIZE);
    }

    void prefetch_vector(NodeId id) const {
        const char* base = reinterpret_cast<const char*>(raw_vectors_[id].data());
        constexpr size_t LINES = std::min((D * sizeof(float) + CACHE_LINE_SIZE - 1) / CACHE_LINE_SIZE, 2 * ((32 * sizeof(float) + CACHE_LINE_SIZE - 1) / CACHE_LINE_SIZE));
        for (size_t line = 0; line < LINES; ++line) prefetch_t<1>(base + line * CACHE_LINE_SIZE);
    }

    void reorder_bfs(NodeId entry) {
        size_t n = search_data_.size();
        std::vector<NodeId> perm(n, INVALID_NODE), new_to_old(n);
        std::vector<bool> visited(n, false);
        std::queue<NodeId> bfs_queue;
        NodeId next_new_id = 0;

        auto run_bfs = [&](NodeId start) {
            if (visited[start]) return;
            bfs_queue.push(start);
            visited[start] = true;
            while (!bfs_queue.empty()) {
                NodeId curr = bfs_queue.front(); bfs_queue.pop();
                perm[curr] = next_new_id;
                new_to_old[next_new_id++] = curr;
                const auto& nb = search_data_[curr].neighbors;
                for (size_t i = 0; i < nb.size(); ++i) {
                    NodeId neighbor = nb.neighbor_ids[i];
                    if (neighbor != INVALID_NODE && !visited[neighbor]) { visited[neighbor] = true; bfs_queue.push(neighbor); }
                }
            }
        };

        run_bfs(entry);
        for (size_t i = 0; i < n; ++i) if (!visited[i]) run_bfs(static_cast<NodeId>(i));

        std::vector<SearchDataType, AlignedAllocator<SearchDataType>> new_search(n);
        std::vector<RawVector> new_vectors(n);
        AlignedVector<float> new_norms(n);
        for (size_t new_id = 0; new_id < n; ++new_id) {
            NodeId old_id = new_to_old[new_id];
            new_search[new_id] = std::move(search_data_[old_id]);
            new_vectors[new_id] = raw_vectors_[old_id];
            new_norms[new_id] = norm_sq_[old_id];
        }
        for (size_t i = 0; i < n; ++i) {
            auto& nb = new_search[i].neighbors;
            for (size_t j = 0; j < GRAPH_DEGREE; ++j)
                if (nb.neighbor_ids[j] != INVALID_NODE) nb.neighbor_ids[j] = perm[nb.neighbor_ids[j]];
        }
        search_data_ = std::move(new_search); raw_vectors_ = std::move(new_vectors); norm_sq_ = std::move(new_norms);
        entry_point_ = perm[entry_point_];
    }

    NodeId find_hub_entry(const float* centroid) const {
        size_t n = raw_vectors_.size();
        size_t top_k = std::max<size_t>(1, static_cast<size_t>(std::sqrt(static_cast<double>(n))));

        std::vector<std::pair<float, NodeId>> dists(n);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i) dists[i] = {l2_distance_simd<D>(raw_vectors_[i].data(), centroid), static_cast<NodeId>(i)};

        std::partial_sort(dists.begin(), dists.begin() + top_k, dists.end());

        NodeId best = dists[0].second;
        size_t best_degree = search_data_[best].neighbors.size();
        for (size_t i = 1; i < top_k; ++i) {
            size_t deg = search_data_[dists[i].second].neighbors.size();
            if (deg > best_degree) { best_degree = deg; best = dists[i].second; }
        }
        return best;
    }
};

}
