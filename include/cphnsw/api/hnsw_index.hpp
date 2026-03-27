#pragma once

#include "../core/core.hpp"
#include "../distance/fastscan_kernel.hpp"
#include "../encoder/rabitq_encoder.hpp"
#include "../graph/rabitq_graph.hpp"
#include "../graph/graph_refinement.hpp"
#include "../search/rabitq_search.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <mutex>
#include <shared_mutex>
#include <cstring>
#include <fstream>
#include <string>

#include <omp.h>

namespace cphnsw {

struct CalibrationSnapshot {
    float gamma_eff;
    float dot_slack;
};

template <size_t D, size_t BitWidth = 1>
class Index {
public:
    using CodeType = NbitRaBitQCode<D, BitWidth>;
    using QueryType = RaBitQQuery<D>;
    using Encoder = NbitRaBitQEncoder<D, BitWidth>;
    using Graph = RaBitQGraph<D, BitWidth>;
    static constexpr size_t DIMS = D;
    static constexpr size_t BIT_WIDTH = BitWidth;

    explicit Index(size_t dim)
        : dim_(dim)
        , encoder_(dim, 42)
        , graph_(dim)
    {}

    void build(const float* vecs, size_t num_vecs) {
        std::unique_lock<std::shared_mutex> lock(index_mutex_);

        graph_ = Graph(dim_);
        calibration_ = {};

        graph_.reserve(num_vecs);

        std::vector<CodeType> codes(num_vecs);
        encoder_.encode_batch(vecs, num_vecs, codes.data());

        for (size_t i = 0; i < num_vecs; ++i) {
            graph_.add_node(codes[i], vecs + i * dim_);
        }
    }

    void finalize() {
        std::unique_lock<std::shared_mutex> lock(index_mutex_);
        graph_refinement::optimize_graph_adaptive(graph_, encoder_);
        compute_analytic_bounds();
    }

    std::vector<SearchResult> search(
        const float* query,
        size_t k) const
    {
        std::shared_lock<std::shared_mutex> lock(index_mutex_);

        thread_local AlignedVector<float> query_padded;
        query_padded.resize(D);
        std::memcpy(query_padded.data(), query, dim_ * sizeof(float));
        if (dim_ < D) {
            std::memset(query_padded.data() + dim_, 0, (D - dim_) * sizeof(float));
        }
        const float* query_vec = query_padded.data();

        QueryType encoded = encoder_.encode_query_raw(query_vec);
        encoded.dot_slack = calibration_.dot_slack;
        float gamma = 1.0f + calibration_.gamma_eff;

        thread_local TwoLevelVisitationTable visited(0);
        if (visited.capacity() < graph_.size()) {
            visited.resize(graph_.size());
        }

        NodeId ep = graph_.entry_point();
        return rabitq_search::search<D, BitWidth>(
            encoded, query_vec, graph_, k, gamma, visited, ep);
    }

    size_t size() const { return graph_.size(); }
    size_t dim() const { return dim_; }

    void save(const std::string& path) const {
        std::shared_lock<std::shared_mutex> lock(index_mutex_);
        std::ofstream f(path, std::ios::binary);

        auto write_raw = [&](const void* data, size_t bytes) {
            f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(bytes));
        };

        uint64_t hdr_n = static_cast<uint64_t>(graph_.size());
        write_raw(&hdr_n, sizeof(hdr_n));
        write_raw(&calibration_, sizeof(CalibrationSnapshot));

        const auto& centroid = encoder_.get_centroid();
        write_raw(centroid.data(), dim_ * sizeof(float));

        size_t n = static_cast<size_t>(hdr_n);

        const auto& ns = graph_.get_norm_sq();
        write_raw(ns.data(), n * sizeof(float));

        const auto& rv = graph_.get_raw_vectors();
        for (size_t i = 0; i < n; ++i) {
            write_raw(rv[i].data(), D * sizeof(float));
        }

        const auto& sd = graph_.get_search_data();
        for (size_t i = 0; i < n; ++i) {
            write_raw(&sd[i], sizeof(typename Graph::SearchDataType));
        }
    }

    void load(const std::string& path) {
        std::unique_lock<std::shared_mutex> lock(index_mutex_);
        std::ifstream f(path, std::ios::binary);

        auto read_raw = [&](void* data, size_t bytes) {
            f.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(bytes));
        };

        uint64_t hdr_n;
        read_raw(&hdr_n, sizeof(hdr_n));

        CalibrationSnapshot new_calib;
        read_raw(&new_calib, sizeof(CalibrationSnapshot));

        size_t n = static_cast<size_t>(hdr_n);

        std::vector<float> new_centroid(dim_);
        read_raw(new_centroid.data(), dim_ * sizeof(float));

        AlignedVector<float> new_norm_sq(n);
        read_raw(new_norm_sq.data(), n * sizeof(float));

        using RawVector = typename Graph::RawVector;
        std::vector<RawVector> new_raw_vectors(n);
        for (size_t i = 0; i < n; ++i) {
            read_raw(new_raw_vectors[i].data(), D * sizeof(float));
        }

        using SearchDataType = typename Graph::SearchDataType;
        std::vector<SearchDataType, AlignedAllocator<SearchDataType>> new_search_data(n);
        for (size_t i = 0; i < n; ++i) {
            read_raw(&new_search_data[i], sizeof(SearchDataType));
        }

        graph_.restore_from_serialized(
            std::move(new_search_data),
            std::move(new_raw_vectors),
            std::move(new_norm_sq),
            graph_.entry_point());

        encoder_.set_centroid(std::move(new_centroid));
        calibration_ = new_calib;
    }

private:
    size_t dim_;
    Encoder encoder_;
    Graph graph_;
    CalibrationSnapshot calibration_;
    mutable std::shared_mutex index_mutex_;

    void compute_analytic_bounds() {
        float sigma_ratio = std::sqrt(static_cast<float>(M_PI) / 2.0f - 1.0f);
        float sqrt_D = std::sqrt(static_cast<float>(D));
        float epsilon_B = sigma_ratio / (static_cast<float>(1 << BitWidth) * sqrt_D);
        float gamma = 2.0f * (1.0f - epsilon_B);
        calibration_.gamma_eff = (1.0f + gamma) * (1.0f + epsilon_B) / (1.0f - epsilon_B) - 1.0f;
        if constexpr (BitWidth >= 2) {
            calibration_.dot_slack = sigma_ratio / sqrt_D;
        } else {
            calibration_.dot_slack = 0.0f;
        }
    }
};

}
