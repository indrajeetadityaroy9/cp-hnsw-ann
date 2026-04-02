#pragma once

#include "core.hpp"
#include "calibration.hpp"
#include "encoder.hpp"
#include "graph.hpp"
#include "graph_build.hpp"
#include "search.hpp"
#include <algorithm>
#include <fstream>
#include <mutex>
#include <shared_mutex>
#include <span>
#include <string>
#include <vector>

#include <omp.h>

namespace evtq {

struct IndexBase {
    virtual ~IndexBase() = default;
    virtual void build(std::span<const float> vecs, size_t num_vecs) = 0;
    virtual void finalize(size_t k, float target_recall) = 0;
    virtual std::vector<SearchResult> search(std::span<const float> query, size_t k) const = 0;
    virtual size_t size() const = 0;
    virtual size_t dim() const = 0;
    virtual void save(const std::string& path) const = 0;
    virtual void load(const std::string& path) = 0;
};

template <size_t D>
struct Index : IndexBase {
    using CodeType = NbitRaBitQCode<D>;
    using QueryType = RaBitQQuery<D>;
    using Encoder = NbitRaBitQEncoder<D>;
    using Graph = RaBitQGraph<D>;
    static constexpr size_t DIMS = D;

    explicit Index(size_t dim)
        : dim_(dim)
        , encoder_(dim, static_cast<uint64_t>(dim) * D)
        , graph_(dim)
    {}

    void build(std::span<const float> vecs, size_t num_vecs) override {
        std::unique_lock<std::shared_mutex> lock(index_mutex_);

        graph_ = Graph(dim_);

        graph_.reserve(num_vecs);

        std::vector<CodeType> codes(num_vecs);
        encoder_.encode_batch(vecs.data(), num_vecs, codes.data());

        for (size_t i = 0; i < num_vecs; ++i) {
            graph_.add_node(codes[i], vecs.subspan(i * dim_, dim_));
        }
    }

    void finalize(size_t k, float target_recall) override {
        std::unique_lock<std::shared_mutex> lock(index_mutex_);
        graph_refinement::optimize_graph_adaptive(graph_, encoder_);
        calibration_ = qrct::calibrate<D>(graph_, encoder_, k, target_recall);
    }

    std::vector<SearchResult> search(
        std::span<const float> query,
        size_t k) const override
    {
        std::shared_lock<std::shared_mutex> lock(index_mutex_);

        thread_local AlignedVector<float> query_padded;
        query_padded.resize(D);
        std::copy_n(query.data(), dim_, query_padded.data());
        if (dim_ < D) {
            std::fill_n(query_padded.data() + dim_, D - dim_, 0.0f);
        }
        const float* query_vec = query_padded.data();

        QueryType encoded = encoder_.encode_query_raw(query_vec);

        thread_local TwoLevelVisitationTable visited(0);
        if (visited.capacity() < graph_.size()) {
            visited.resize(graph_.size());
        }

        NodeId ep = graph_.entry_point();
        return rabitq_search::search<D>(
            encoded, query_vec, graph_, k, visited, ep, calibration_);
    }

    size_t size() const override { return graph_.size(); }
    size_t dim() const override { return dim_; }

    void save(const std::string& path) const override {
        std::shared_lock<std::shared_mutex> lock(index_mutex_);
        std::ofstream f(path, std::ios::binary);

        auto write_raw = [&](const void* data, size_t bytes) {
            f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(bytes));
        };

        uint64_t hdr_n = static_cast<uint64_t>(graph_.size());
        write_raw(&hdr_n, sizeof(hdr_n));
        NodeId ep = graph_.entry_point();
        write_raw(&ep, sizeof(ep));
        write_raw(&calibration_, sizeof(QRCTCalibration));
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

    void load(const std::string& path) override {
        std::unique_lock<std::shared_mutex> lock(index_mutex_);
        std::ifstream f(path, std::ios::binary);

        auto read_raw = [&](void* data, size_t bytes) {
            f.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(bytes));
        };

        uint64_t hdr_n;
        read_raw(&hdr_n, sizeof(hdr_n));
        NodeId ep;
        read_raw(&ep, sizeof(ep));
        read_raw(&calibration_, sizeof(QRCTCalibration));

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
    }

    size_t dim_;
    Encoder encoder_;
    Graph graph_;
    QRCTCalibration calibration_;
    mutable std::shared_mutex index_mutex_;
};

}
