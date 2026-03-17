#pragma once

#include "../core/core.hpp"
#include "../core/evt_crc.hpp"
#include "../distance/fastscan_kernel.hpp"
#include "../encoder/rabitq_encoder.hpp"
#include "../graph/rabitq_graph.hpp"
#include "../graph/graph_refinement.hpp"
#include "../search/rabitq_search.hpp"
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <mutex>
#include <shared_mutex>
#include <limits>
#include <cstring>
#include <fstream>
#include <string>

#include <omp.h>

namespace cphnsw {

struct CalibrationSnapshot {
    float affine_a;
    float affine_b;
    float ip_qo_floor;

    float search_ip_slack_levels[8];
    int search_num_slack_levels;
    float search_gamma;
};


struct HNSWLayerEdge {
    NodeId node;
    std::vector<NodeId> neighbors;
    bool operator<(const HNSWLayerEdge& other) const { return node < other.node; }
};

template <size_t D, size_t R = 32, size_t BitWidth = 1>
class Index {
public:
    using CodeType = NbitRaBitQCode<D, BitWidth>;
    using QueryType = RaBitQQuery<D>;
    using Encoder = NbitRaBitQEncoder<D, BitWidth>;
    using Graph = RaBitQGraph<D, R, BitWidth>;
    static constexpr size_t DIMS = D;
    static constexpr size_t DEGREE = R;
    static constexpr size_t BIT_WIDTH = BitWidth;

    static constexpr size_t M_UPPER = R / 2;

    explicit Index(size_t dim)
        : dim_(dim)
        , encoder_(dim, 42)
        , graph_(dim)
        , mL_(1.0 / std::log(static_cast<double>(M_UPPER)))
        , rng_(42)
    {}

    void build(const float* vecs, size_t num_vecs) {
        std::unique_lock<std::shared_mutex> lock(index_mutex_);

        graph_ = Graph(dim_);
        calibration_ = {};
        max_level_ = 0;
        entry_point_ = INVALID_NODE;
        node_levels_.clear();
        upper_layers_.clear();

        graph_.reserve(num_vecs);

        std::vector<CodeType> codes(num_vecs);
        encoder_.encode_batch(vecs, num_vecs, codes.data());

        for (size_t i = 0; i < num_vecs; ++i) {
            graph_.add_node(codes[i], vecs + i * dim_);
        }
    }

    void finalize() {
        std::unique_lock<std::shared_mutex> lock(index_mutex_);
        size_t n = graph_.size();

        size_t evt_min_tail = static_cast<size_t>(std::sqrt(static_cast<double>(n)));

        size_t min_calib_samples = n / R;

        float log_n = std::log2(static_cast<float>(n));
        int slack_levels = std::clamp(
            static_cast<int>(std::ceil(std::log2(log_n))),
            1, 8);

        assign_layers(n);

        build_upper_layers();

        auto perm = graph_refinement::optimize_graph_adaptive(
            graph_, encoder_);

        for (auto& layer : upper_layers_) {
            for (auto& edge : layer) {
                edge.node = perm[edge.node];
                for (auto& nb : edge.neighbors) {
                    nb = perm[nb];
                }
            }
            std::sort(layer.begin(), layer.end());
        }
        entry_point_ = perm[entry_point_];
        std::vector<int> new_levels(node_levels_.size());
        for (size_t i = 0; i < node_levels_.size(); ++i) {
            new_levels[perm[i]] = node_levels_[i];
        }
        node_levels_ = std::move(new_levels);

        size_t n_calib = std::min(min_calib_samples, n);
        calibrate_estimator(n_calib, evt_min_tail, slack_levels);
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
        encoded.affine_a = calibration_.affine_a;
        encoded.affine_b = calibration_.affine_b;
        encoded.ip_qo_floor = calibration_.ip_qo_floor;
        encoded.dot_slack = calibration_.search_ip_slack_levels[0];

        float gamma = calibration_.search_gamma;

        thread_local TwoLevelVisitationTable visited(0);
        if (visited.capacity() < graph_.size()) {
            visited.resize(graph_.size());
        }

        NodeId ep = graph_.entry_point();
        if (max_level_ > 0) {
            ep = entry_point_;
            for (int level = max_level_; level >= 1; --level) {
                ep = greedy_search_layer(query_vec, ep, level);
            }
        }
        return rabitq_search::search<D, R, BitWidth>(
            encoded, query_vec, graph_, k, gamma, visited, ep,
            calibration_.search_ip_slack_levels, calibration_.search_num_slack_levels);
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
        int32_t hdr_max_level = static_cast<int32_t>(max_level_);
        uint32_t hdr_ep = static_cast<uint32_t>(entry_point_);

        write_raw(&hdr_n, sizeof(hdr_n));
        write_raw(&hdr_max_level, sizeof(hdr_max_level));
        write_raw(&hdr_ep, sizeof(hdr_ep));
        write_raw(&mL_, sizeof(mL_));

        write_raw(&calibration_, sizeof(CalibrationSnapshot));

        const auto& centroid = encoder_.get_centroid();
        write_raw(centroid.data(), dim_ * sizeof(float));

        size_t n = static_cast<size_t>(hdr_n);
        write_raw(node_levels_.data(), n * sizeof(int));

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

        uint32_t n_layers = static_cast<uint32_t>(upper_layers_.size());
        write_raw(&n_layers, sizeof(n_layers));
        for (const auto& layer : upper_layers_) {
            uint32_t layer_sz = static_cast<uint32_t>(layer.size());
            write_raw(&layer_sz, sizeof(layer_sz));
            for (const auto& edge : layer) {
                write_raw(&edge.node, sizeof(edge.node));
                uint32_t nb_cnt = static_cast<uint32_t>(edge.neighbors.size());
                write_raw(&nb_cnt, sizeof(nb_cnt));
                write_raw(edge.neighbors.data(), nb_cnt * sizeof(NodeId));
            }
        }
    }

    void load(const std::string& path) {
        std::unique_lock<std::shared_mutex> lock(index_mutex_);
        std::ifstream f(path, std::ios::binary);

        auto read_raw = [&](void* data, size_t bytes) {
            f.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(bytes));
        };

        uint64_t hdr_n;
        int32_t hdr_max_level;
        uint32_t hdr_ep;
        double hdr_mL;

        read_raw(&hdr_n, sizeof(hdr_n));
        read_raw(&hdr_max_level, sizeof(hdr_max_level));
        read_raw(&hdr_ep, sizeof(hdr_ep));
        read_raw(&hdr_mL, sizeof(hdr_mL));

        CalibrationSnapshot new_calib;
        read_raw(&new_calib, sizeof(CalibrationSnapshot));

        size_t n = static_cast<size_t>(hdr_n);

        std::vector<float> new_centroid(dim_);
        read_raw(new_centroid.data(), dim_ * sizeof(float));

        std::vector<int> new_node_levels(n);
        read_raw(new_node_levels.data(), n * sizeof(int));

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

        uint32_t n_layers;
        read_raw(&n_layers, sizeof(n_layers));
        std::vector<std::vector<HNSWLayerEdge>> new_upper_layers(n_layers);
        for (uint32_t l = 0; l < n_layers; ++l) {
            uint32_t layer_sz;
            read_raw(&layer_sz, sizeof(layer_sz));
            new_upper_layers[l].resize(layer_sz);
            for (uint32_t e = 0; e < layer_sz; ++e) {
                read_raw(&new_upper_layers[l][e].node, sizeof(NodeId));
                uint32_t nb_cnt;
                read_raw(&nb_cnt, sizeof(nb_cnt));
                new_upper_layers[l][e].neighbors.resize(nb_cnt);
                read_raw(new_upper_layers[l][e].neighbors.data(),
                         nb_cnt * sizeof(NodeId));
            }
        }

        graph_.restore_from_serialized(
            std::move(new_search_data),
            std::move(new_raw_vectors),
            std::move(new_norm_sq),
            static_cast<NodeId>(hdr_ep));

        encoder_.set_centroid(std::move(new_centroid));
        calibration_ = new_calib;
        node_levels_ = std::move(new_node_levels);
        upper_layers_ = std::move(new_upper_layers);
        max_level_ = static_cast<int>(hdr_max_level);
        entry_point_ = static_cast<NodeId>(hdr_ep);
        mL_ = hdr_mL;
    }

private:
    size_t dim_;
    Encoder encoder_;
    Graph graph_;

    CalibrationSnapshot calibration_;

    double mL_;
    std::mt19937_64 rng_;

    int max_level_ = 0;
    NodeId entry_point_ = INVALID_NODE;

    std::vector<int> node_levels_;

    std::vector<std::vector<HNSWLayerEdge>> upper_layers_;
    mutable std::shared_mutex index_mutex_;

    const HNSWLayerEdge* find_edge(int level, NodeId node) const {
        const auto& layer = upper_layers_[level - 1];
        HNSWLayerEdge target{node, {}};
        auto it = std::lower_bound(layer.begin(), layer.end(), target);
        if (it != layer.end() && it->node == node) return &(*it);
        return nullptr;
    }

    HNSWLayerEdge& get_or_create_edge(int level, NodeId node) {
        auto& layer = upper_layers_[level - 1];
        HNSWLayerEdge target{node, {}};
        auto it = std::lower_bound(layer.begin(), layer.end(), target);
        if (it != layer.end() && it->node == node) return *it;
        return *layer.insert(it, HNSWLayerEdge{node, {}});
    }

    void assign_layers(size_t n) {
        node_levels_.resize(n);
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        max_level_ = 0;
        entry_point_ = INVALID_NODE;

        for (size_t i = 0; i < n; ++i) {
            double r = dist(rng_);
            if (r < static_cast<double>(std::numeric_limits<float>::epsilon())) r = static_cast<double>(std::numeric_limits<float>::epsilon());
            int level = static_cast<int>(-std::log(r) * mL_);
            node_levels_[i] = level;
            if (entry_point_ == INVALID_NODE || level > max_level_) {
                max_level_ = level;
                entry_point_ = static_cast<NodeId>(i);
            }
        }

        upper_layers_.resize(max_level_);
    }

    void build_upper_layers() {
        size_t n = graph_.size();

        std::vector<NodeId> insertion_order(n);
        for (size_t i = 0; i < n; ++i) insertion_order[i] = static_cast<NodeId>(i);
        std::sort(insertion_order.begin(), insertion_order.end(),
                  [this](NodeId a, NodeId b) { return node_levels_[a] > node_levels_[b]; });

        size_t n_upper = 0;
        for (size_t i = 0; i < n; ++i) {
            if (node_levels_[insertion_order[i]] > 0) n_upper++;
            else break;
        }

        std::vector<float> upper_nn_dists;
        for (size_t idx = 0; idx < n; ++idx) {
            NodeId node = insertion_order[idx];
            if (node_levels_[node] == 0) break;
            float best = std::numeric_limits<float>::max();
            for (size_t jdx = 0; jdx < n; ++jdx) {
                NodeId other = insertion_order[jdx];
                if (other == node) continue;
                if (node_levels_[other] == 0) break;
                float d = l2_distance_simd<D>(graph_.get_vector(node), graph_.get_vector(other));
                if (d < best) best = d;
            }
            if (best < std::numeric_limits<float>::max()) {
                upper_nn_dists.push_back(best);
            }
        }
        std::sort(upper_nn_dists.begin(), upper_nn_dists.end());
        float upper_tau = stddev(upper_nn_dists.data(), upper_nn_dists.size());

        float mean_dist = 0.0f;
        for (float d : upper_nn_dists) mean_dist += d;
        mean_dist /= static_cast<float>(upper_nn_dists.size());
        float var_dist = 0.0f;
        for (float d : upper_nn_dists) var_dist += (d - mean_dist) * (d - mean_dist);
        var_dist /= static_cast<float>(upper_nn_dists.size());
        float upper_alpha = (mean_dist > std::numeric_limits<float>::epsilon())
            ? 1.0f + std::sqrt(var_dist) / mean_dist : 1.0f;

        for (size_t idx = 0; idx < n; ++idx) {
            NodeId node = insertion_order[idx];
            int node_level = node_levels_[node];
            if (node_level == 0) break;

            NodeId ep = entry_point_;
            for (int level = max_level_; level > node_level; --level) {
                ep = greedy_search_layer(graph_.get_vector(node), ep, level);
            }

            for (int level = std::min(node_level, max_level_); level >= 1; --level) {
                size_t upper_ef = std::max(R,
                    static_cast<size_t>(static_cast<float>(R) *
                        (1.0f + static_cast<float>(level) *
                         std::log(static_cast<float>(n_upper)) /
                         std::log(static_cast<float>(n)))));
                auto candidates = search_upper_layer(
                    graph_.get_vector(node), ep, level, upper_ef);

                auto dist_fn = [this](NodeId a, NodeId b) {
                    return l2_distance_simd<D>(graph_.get_vector(a), graph_.get_vector(b));
                };
                auto selected = select_neighbors_alpha_cng(
                    std::move(candidates), M_UPPER, dist_fn, [](NodeId) { return 0.0f; },
                    upper_alpha, upper_tau);

                auto& node_neighbors = get_or_create_edge(level, node).neighbors;
                node_neighbors.clear();
                node_neighbors.reserve(selected.size());
                for (const auto& s : selected) {
                    node_neighbors.push_back(s.id);
                }

                for (const auto& s : selected) {
                    auto& nb = get_or_create_edge(level, s.id).neighbors;
                    nb.push_back(node);
                    if (nb.size() > M_UPPER) {
                        prune_upper_neighbors(s.id, level, upper_alpha, upper_tau);
                    }
                }

                ep = selected[0].id;
            }
        }

    }

    NodeId greedy_search_layer(const float* query, NodeId ep, int level) const {
        float best_dist = l2_distance_simd<D>(query, graph_.get_vector(ep));
        NodeId best_id = ep;
        bool improved = true;

        while (improved) {
            improved = false;
            const auto* edge = find_edge(level, best_id);
            if (!edge) break;
            const auto& neighbors = edge->neighbors;
            for (NodeId nb : neighbors) {
                float d = l2_distance_simd<D>(query, graph_.get_vector(nb));
                if (d < best_dist) {
                    best_dist = d;
                    best_id = nb;
                    improved = true;
                }
            }
        }

        return best_id;
    }

    std::vector<SearchResult> search_upper_layer(
        const float* query, NodeId ep, int level, size_t ef) const
    {
        MinHeap candidates;
        MaxHeap nearest;

        float ep_dist = l2_distance_simd<D>(query, graph_.get_vector(ep));
        candidates.push({ep, ep_dist});
        nearest.push({ep, ep_dist});

        thread_local TwoLevelVisitationTable visited_table(0);
        if (visited_table.capacity() < graph_.size()) {
            visited_table.resize(graph_.size());
        }
        uint64_t qid = visited_table.new_query();
        visited_table.check_and_mark_estimated(ep, qid);

        while (!candidates.empty()) {
            auto current = candidates.top();
            candidates.pop();

            if (nearest.size() >= ef && current.distance > nearest.top().distance) {
                break;
            }

            const auto* edge = find_edge(level, current.id);
            if (!edge) continue;
            const auto& neighbors = edge->neighbors;
            for (NodeId nb : neighbors) {
                if (visited_table.check_and_mark_estimated(nb, qid)) continue;

                float d = l2_distance_simd<D>(query, graph_.get_vector(nb));

                if (nearest.size() < ef || d < nearest.top().distance) {
                    candidates.push({nb, d});
                    nearest.push({nb, d});
                    if (nearest.size() > ef) {
                        nearest.pop();
                    }
                }
            }
        }

        std::vector<SearchResult> results;
        results.reserve(nearest.size());
        while (!nearest.empty()) {
            results.push_back({nearest.top().id, nearest.top().distance});
            nearest.pop();
        }
        std::sort(results.begin(), results.end());
        return results;
    }

    void prune_upper_neighbors(NodeId node, int level, float alpha, float tau) {
        auto& nb = get_or_create_edge(level, node).neighbors;
        if (nb.size() <= M_UPPER) return;

        const float* vec = graph_.get_vector(node);
        std::vector<SearchResult> candidates;
        candidates.reserve(nb.size());
        for (NodeId id : nb) {
            float d = l2_distance_simd<D>(vec, graph_.get_vector(id));
            candidates.push_back({id, d});
        }

        auto dist_fn = [this](NodeId a, NodeId b) {
            return l2_distance_simd<D>(graph_.get_vector(a), graph_.get_vector(b));
        };
        auto selected = select_neighbors_alpha_cng(
            std::move(candidates), M_UPPER, dist_fn, [](NodeId) { return 0.0f; },
            alpha, tau);
        nb.clear();
        nb.reserve(selected.size());
        for (const auto& s : selected) {
            nb.push_back(s.id);
        }
    }

    void calibrate_estimator(size_t num_samples, size_t evt_min_tail, int slack_levels) {
        size_t n = graph_.size();

        std::vector<NodeId> sample_ids(n);
        for (size_t i = 0; i < n; ++i) {
            sample_ids[i] = static_cast<NodeId>(i);
        }

        std::mt19937 rng(static_cast<uint32_t>(42));
        std::shuffle(sample_ids.begin(), sample_ids.end(), rng);

        size_t n_db = std::min(num_samples, n);
        size_t n_synth = std::min(num_samples, n);

        std::vector<float> dim_var(D, 0.0f);
        std::vector<float> dim_mean(D, 0.0f);
        for (size_t i = 0; i < n_db; ++i) {
            const float* v = graph_.get_vector(sample_ids[i]);
            for (size_t d = 0; d < D; ++d) {
                dim_mean[d] += v[d];
                dim_var[d] += v[d] * v[d];
            }
        }
        for (size_t d = 0; d < D; ++d) {
            dim_mean[d] /= static_cast<float>(n_db);
            dim_var[d] = dim_var[d] / static_cast<float>(n_db) - dim_mean[d] * dim_mean[d];
            if (dim_var[d] < std::numeric_limits<float>::epsilon()) dim_var[d] = std::numeric_limits<float>::epsilon();
        }


        struct CalibSample {
            float nop;
            float ip_corrected;
            float ip_qo_denom;
            float dist_qp_sq;
            NodeId neighbor;
            size_t query_idx;
        };

        std::vector<float> ip_qo_values;
        std::vector<float> per_sample_ip_corrected;
        std::vector<float> per_sample_ip_qo;
        std::vector<float> truths;
        std::vector<float> nn_dists_sq;
        std::vector<float> nop_samples;
        std::vector<CalibSample> calib_samples;


        std::vector<AlignedVector<float>> query_buffer;
        query_buffer.reserve(n_db + n_synth);

        ip_qo_values.reserve(num_samples * 4);
        per_sample_ip_corrected.reserve(num_samples * 4);
        per_sample_ip_qo.reserve(num_samples * 4);
        truths.reserve(num_samples * 4);
        nop_samples.reserve(num_samples * 4);
        calib_samples.reserve(num_samples * 4);

        size_t parent_cursor = 0;
        auto process_query = [&](const float* query_vec, size_t query_idx) {
            NodeId parent = sample_ids[parent_cursor % n];
            parent_cursor++;

            float best_dist = l2_distance_simd<D>(query_vec, graph_.get_vector(parent));
            const auto& nb = graph_.get_neighbors(parent);

            for (size_t i = 0; i < nb.size(); ++i) {
                NodeId nid = nb.neighbor_ids[i];
                if (nid == INVALID_NODE) break;
                float d = l2_distance_simd<D>(query_vec, graph_.get_vector(nid));
                if (d < best_dist) {
                    best_dist = d;
                    parent = nid;
                }
            }
            nn_dists_sq.push_back(best_dist);

            const auto& pnb = graph_.get_neighbors(parent);

            QueryType encoded = encoder_.encode_query_raw(query_vec);

            float dist_qp_sq = l2_distance_simd<D>(query_vec, graph_.get_vector(parent));

            size_t num_batches = (pnb.size() + fastscan::BATCH_SIZE - 1) / fastscan::BATCH_SIZE;
            for (size_t batch = 0; batch < num_batches; ++batch) {
                size_t batch_start = batch * fastscan::BATCH_SIZE;
                size_t batch_count = std::min(fastscan::BATCH_SIZE, pnb.size() - batch_start);

                alignas(64) uint32_t fastscan_sums[fastscan::BATCH_SIZE];
                alignas(64) uint32_t msb_sums[fastscan::BATCH_SIZE];
                fastscan::compute_nbit_inner_products<D, BitWidth>(
                    encoded.lut, pnb.code_blocks[batch],
                    fastscan_sums, msb_sums);

                for (size_t j = 0; j < batch_count; ++j) {
                    size_t ni = batch_start + j;
                    NodeId neighbor = pnb.neighbor_ids[ni];
                    if (neighbor == INVALID_NODE) break;

                    float ip_qo = pnb.ip_qo[ni];
                    ip_qo_values.push_back(ip_qo);

                    float nop = std::max(pnb.nop[ni], std::numeric_limits<float>::epsilon());
                    nop_samples.push_back(nop);

                    float A = encoded.coeff_fastscan;
                    float B = encoded.coeff_popcount;
                    float C = encoded.coeff_constant;

                    constexpr float K = static_cast<float>((1u << BitWidth) - 1);
                    constexpr float inv_K = 1.0f / K;
                    float ip_approx = A * inv_K * static_cast<float>(fastscan_sums[j])
                                    + B * inv_K * static_cast<float>(pnb.weighted_popcounts[ni]) + C;

                    float ip_corrected = ip_approx - pnb.ip_cp[ni];
                    float ip_qo_denom = std::max(std::abs(ip_qo), std::numeric_limits<float>::epsilon());

                    const float* p_vec = graph_.get_vector(parent);
                    const float* o_vec = graph_.get_vector(neighbor);
                    float true_ip = 0.0f;
                    for (size_t d = 0; d < D; ++d) {
                        true_ip += (query_vec[d] - p_vec[d]) * (o_vec[d] - p_vec[d]);
                    }
                    true_ip /= nop;

                    per_sample_ip_corrected.push_back(ip_corrected);
                    per_sample_ip_qo.push_back(ip_qo_denom);
                    truths.push_back(true_ip);

                    calib_samples.push_back({nop, ip_corrected, ip_qo_denom,
                                             dist_qp_sq, neighbor, query_idx});
                }
            }
        };

        for (size_t i = 0; i < n_db; ++i) {
            const float* v = graph_.get_vector(sample_ids[i]);
            AlignedVector<float> qbuf(D);
            std::memcpy(qbuf.data(), v, D * sizeof(float));
            query_buffer.push_back(std::move(qbuf));
            process_query(v, query_buffer.size() - 1);
        }

        std::normal_distribution<float> normal_dist(0.0f, 1.0f);
        for (size_t i = 0; i < n_synth; ++i) {
            const float* base = graph_.get_vector(sample_ids[i % n]);
            AlignedVector<float> synth_query(D);
            for (size_t d = 0; d < D; ++d) {
                synth_query[d] = base[d] + normal_dist(rng) * std::sqrt(dim_var[d]);
            }
            query_buffer.push_back(synth_query);
            process_query(query_buffer.back().data(), query_buffer.size() - 1);
        }


        float sigma_ipqo = stddev(ip_qo_values.data(), ip_qo_values.size());
        float mean_ipqo = 0.0f;
        for (float v : ip_qo_values) mean_ipqo += v;
        mean_ipqo /= static_cast<float>(ip_qo_values.size());
        calibration_.ip_qo_floor = std::max(
            mean_ipqo - 3.0f * sigma_ipqo,
            std::numeric_limits<float>::epsilon());

        std::vector<float> floored_estimates;
        floored_estimates.reserve(per_sample_ip_corrected.size());
        for (size_t i = 0; i < per_sample_ip_corrected.size(); ++i) {
            float floored_qo = std::max(per_sample_ip_qo[i], calibration_.ip_qo_floor);
            floored_estimates.push_back(per_sample_ip_corrected[i] / floored_qo);
        }


        size_t np = floored_estimates.size();
        double sum_e = 0, sum_t = 0, sum_ee = 0, sum_et = 0;
        for (size_t i = 0; i < np; ++i) {
            double e = floored_estimates[i];
            double t = truths[i];
            sum_e += e;
            sum_t += t;
            sum_ee += e * e;
            sum_et += e * t;
        }
        double mean_e = sum_e / np;
        double mean_t = sum_t / np;
        double var_e = sum_ee / np - mean_e * mean_e;
        double cov_et = sum_et / np - mean_e * mean_t;

        double a = 1.0, b = 0.0;
        if (var_e > std::numeric_limits<float>::epsilon()) {
            a = cov_et / var_e;
            b = mean_t - a * mean_e;
        }

        std::vector<float> abs_residuals(np);
        for (;;) {
            for (size_t i = 0; i < np; ++i) {
                float r = truths[i] - static_cast<float>(a * floored_estimates[i] + b);
                abs_residuals[i] = std::abs(r);
            }
            float resid_sigma = stddev(abs_residuals.data(), np);
            float huber_delta = 1.345f * resid_sigma /* Huber 95% efficiency */;
            if (huber_delta < std::numeric_limits<float>::epsilon()) break;

            double wsum_e = 0, wsum_t = 0, wsum_ee = 0, wsum_et = 0, wsum = 0;
            for (size_t i = 0; i < np; ++i) {
                float r = truths[i] - static_cast<float>(a * floored_estimates[i] + b);
                float ar = std::abs(r);
                float w = (ar <= huber_delta) ? 1.0f : huber_delta / ar;
                double wd = w;
                double e = floored_estimates[i];
                double t = truths[i];
                wsum += wd;
                wsum_e += wd * e;
                wsum_t += wd * t;
                wsum_ee += wd * e * e;
                wsum_et += wd * e * t;
            }
            double wm_e = wsum_e / wsum;
            double wm_t = wsum_t / wsum;
            double wvar = wsum_ee / wsum - wm_e * wm_e;
            double wcov = wsum_et / wsum - wm_e * wm_t;
            if (wvar > std::numeric_limits<float>::epsilon()) {
                double a_new = wcov / wvar;
                double b_new = wm_t - a_new * wm_e;
                if (std::abs(a_new - a) + std::abs(b_new - b) < std::numeric_limits<float>::epsilon()) {
                    a = a_new;
                    b = b_new;
                    break;
                }
                a = a_new;
                b = b_new;
            }
        }


        calibration_.affine_a = static_cast<float>(a);
        calibration_.affine_b = static_cast<float>(b);

        std::sort(nn_dists_sq.begin(), nn_dists_sq.end());
        float median_nn_dist_sq = nn_dists_sq[nn_dists_sq.size() / 2];

        std::vector<float> dist_residuals;
        dist_residuals.reserve(calib_samples.size());
        for (const auto& s : calib_samples) {
            float floored_qo = std::max(s.ip_qo_denom, calibration_.ip_qo_floor);
            float ip_est = (floored_qo > std::numeric_limits<float>::epsilon())
                ? s.ip_corrected / floored_qo : 0.0f;
            ip_est = calibration_.affine_a * ip_est + calibration_.affine_b;
            float est_dist = std::max(s.nop * s.nop + s.dist_qp_sq
                             - 2.0f * s.nop * ip_est, 0.0f);
            const float* qvec = query_buffer[s.query_idx].data();
            float true_dist = l2_distance_simd<D>(qvec, graph_.get_vector(s.neighbor));
            dist_residuals.push_back(std::abs(est_dist - true_dist));
        }

        std::sort(dist_residuals.begin(), dist_residuals.end());

        size_t n_resid = dist_residuals.size();

        EVTState evt = evt_crc::fit_gpd_stable(
            dist_residuals.data(), n_resid, evt_min_tail);

        std::sort(nop_samples.begin(), nop_samples.end());
        float median_nop = nop_samples[nop_samples.size() / 2];

        float ref = std::sqrt(std::max(median_nn_dist_sq, std::numeric_limits<float>::epsilon()));

        float gamma_min = std::max(
            1.0f + dist_residuals[n_resid / 100] / ref,
            1.0f + 1.0f / std::sqrt(static_cast<float>(D)));

        float delta = 1.0f / static_cast<float>(n);

        int evt_L = std::clamp(slack_levels, 1, 8);
        calibration_.search_num_slack_levels = evt_L;

        for (int i = 1; i <= evt_L; ++i) {
            float i_f = static_cast<float>(i);
            float alpha_i = delta * (6.0f / (3.14159265358979f * 3.14159265358979f)) /* 6/π² */ / (i_f * i_f);
            float dist_slack = evt_crc::evt_quantile(alpha_i, evt);
            calibration_.search_ip_slack_levels[i - 1] = dist_slack
                / (2.0f * median_nop);
        }

        float dist_slack_term = evt_crc::evt_quantile(delta, evt);
        calibration_.search_gamma = std::max(
            gamma_min, 1.0f + dist_slack_term / ref);

    }
};

}
