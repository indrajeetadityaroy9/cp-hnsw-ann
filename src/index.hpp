#pragma once

#include "core.hpp"
#include "calibration.hpp"
#include "encoder.hpp"
#include "graph.hpp"
#include "graph_build.hpp"
#include "search.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <span>
#include <string>
#include <vector>
#include <omp.h>

namespace evtq {

struct IndexBase {
    virtual ~IndexBase() = default;
    virtual void build(std::span<const float> vecs, size_t num_vecs) = 0;
    virtual void finalize() = 0;
    virtual std::vector<SearchResult> search(std::span<const float> query, size_t k) const = 0;
    virtual size_t size() const = 0;
    virtual size_t dim() const = 0;
    virtual void save(const std::string& path) const = 0;
    virtual void load(const std::string& path) = 0;
};

template <size_t D>
struct Index : IndexBase {
    size_t dim_;
    NbitRaBitQEncoder<D> encoder_;
    RaBitQGraph<D> graph_;
    GPDCalibration calibration_;

    explicit Index(size_t dim) : dim_(dim), encoder_(dim, static_cast<uint64_t>(dim) * D), graph_(dim) {}

    void build(std::span<const float> vecs, size_t num_vecs) override {
        graph_ = RaBitQGraph<D>(dim_);
        graph_.reserve(num_vecs);
        std::vector<NbitRaBitQCode<D>> codes(num_vecs);
        encoder_.encode_batch(vecs.data(), num_vecs, codes.data());
        for (size_t i = 0; i < num_vecs; ++i) graph_.add_node(codes[i], vecs.subspan(i * dim_, dim_));
    }

    void finalize() override {
        graph_refinement::rnn_descent_build(graph_, encoder_);
        size_t num_probes = static_cast<size_t>(std::ceil(std::sqrt(static_cast<double>(graph_.size()))));
        calibration_ = rcgr::calibrate<D>(graph_, encoder_, num_probes);
    }

    std::vector<SearchResult> search(std::span<const float> query, size_t k) const override {
        alignas(SIMD_ALIGNMENT) float query_padded[D];
        std::copy_n(query.data(), dim_, query_padded);
        std::fill_n(query_padded + dim_, D - dim_, 0.0f);
        auto encoded = encoder_.encode_query_raw(query_padded);
        thread_local VisitationTable visited(0);
        if (visited.capacity() < graph_.size()) visited.resize(graph_.size());
        return rabitq_search::search<D>(encoded, query_padded, graph_, k, visited, graph_.entry_point(), calibration_);
    }

    size_t size() const override { return graph_.size(); }
    size_t dim() const override { return dim_; }

    void save(const std::string& path) const override {
        std::ofstream f(path, std::ios::binary);
        auto write_raw = [&](const void* data, size_t bytes) { f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(bytes)); };

        uint64_t hdr_n = static_cast<uint64_t>(graph_.size());
        NodeId ep = graph_.entry_point();
        write_raw(&hdr_n, sizeof(hdr_n));
        write_raw(&ep, sizeof(ep));
        write_raw(&calibration_, sizeof(GPDCalibration));
        write_raw(encoder_.get_centroid().data(), dim_ * sizeof(float));

        size_t n = static_cast<size_t>(hdr_n);
        write_raw(graph_.get_norm_sq().data(), n * sizeof(float));
        for (size_t i = 0; i < n; ++i) write_raw(graph_.get_raw_vectors()[i].data(), D * sizeof(float));
        for (size_t i = 0; i < n; ++i) write_raw(&graph_.get_search_data()[i], sizeof(VertexSearchData<D>));
    }

    void load(const std::string& path) override {
        std::ifstream f(path, std::ios::binary);
        auto read_raw = [&](void* data, size_t bytes) { f.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(bytes)); };

        uint64_t hdr_n; NodeId ep;
        read_raw(&hdr_n, sizeof(hdr_n));
        read_raw(&ep, sizeof(ep));
        read_raw(&calibration_, sizeof(GPDCalibration));

        size_t n = static_cast<size_t>(hdr_n);
        std::vector<float> new_centroid(dim_);
        read_raw(new_centroid.data(), dim_ * sizeof(float));

        AlignedVector<float> new_norm_sq(n);
        read_raw(new_norm_sq.data(), n * sizeof(float));

        std::vector<std::array<float, D>> new_raw_vectors(n);
        for (size_t i = 0; i < n; ++i) read_raw(new_raw_vectors[i].data(), D * sizeof(float));

        std::vector<VertexSearchData<D>, AlignedAllocator<VertexSearchData<D>>> new_search_data(n);
        for (size_t i = 0; i < n; ++i) read_raw(&new_search_data[i], sizeof(VertexSearchData<D>));

        graph_.restore_from_serialized(std::move(new_search_data), std::move(new_raw_vectors), std::move(new_norm_sq), ep);
        encoder_.set_centroid(std::move(new_centroid));
    }
};

}
