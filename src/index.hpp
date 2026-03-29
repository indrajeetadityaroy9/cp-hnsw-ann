#pragma once

#include "core.hpp"
#include "calibration.hpp"
#include "encoder.hpp"
#include "graph.hpp"
#include "graph_build.hpp"
#include "search.hpp"
#include <vector>
#include <cmath>
#include <mutex>
#include <shared_mutex>
#include <cstring>
#include <fstream>
#include <string>

#include <omp.h>

namespace cphnsw {

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
        , encoder_(dim, static_cast<uint64_t>(dim) * D)
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
        compute_calibration();
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
        NodeId ep = graph_.entry_point();
        write_raw(&ep, sizeof(ep));
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
        NodeId ep;
        read_raw(&ep, sizeof(ep));

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
            ep);

        encoder_.set_centroid(std::move(new_centroid));
        calibration_ = new_calib;
    }

private:
    size_t dim_;
    Encoder encoder_;
    Graph graph_;
    CalibrationSnapshot calibration_;
    mutable std::shared_mutex index_mutex_;

    void compute_calibration() {
        float sigma_ratio = std::sqrt(static_cast<float>(M_PI) / 2.0f - 1.0f);
        float D_f = static_cast<float>(D);

        // Segmentation-aware epsilon:
        // epsilon_seg = sigma_ratio * sqrt(sum_s sum_{j in Seg_s} V_j / 4^{B_s}) / D
        const auto& segments = encoder_.get_segments();
        const auto& var = encoder_.get_dim_variance();
        float weighted_sum = 0.0f;
        float max_seg_weight = 0.0f;
        for (const auto& seg : segments) {
            float scale = 1.0f;
            for (size_t b = 0; b < seg.bits; ++b) scale *= 4.0f;
            float seg_var_sum = 0.0f;
            for (size_t j = seg.start; j < seg.start + seg.len; ++j) {
                seg_var_sum += var[j];
            }
            float w = seg_var_sum / scale;
            weighted_sum += w;
            if (w > max_seg_weight) max_seg_weight = w;
        }

        float epsilon_seg = sigma_ratio * std::sqrt(weighted_sum) / D_f;
        float dot_slack = sigma_ratio * std::sqrt(max_seg_weight) / std::sqrt(D_f);
        
        constexpr float GEOMETRIC_FACTOR = 2.0f;
        constexpr float Z_999 = 3.09f;
        float analytical_gamma_eff = GEOMETRIC_FACTOR * Z_999 * epsilon_seg;

        // GPD calibration with analytical fallback
        calibration_ = calibration::calibrate<D, BitWidth>(
            graph_, encoder_, dot_slack, analytical_gamma_eff);
    }
};

}
