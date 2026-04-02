#pragma once

#include "core.hpp"
#include "encoder.hpp"
#include "estimator.hpp"
#include "graph.hpp"
#include "search.hpp"

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

#include <omp.h>

namespace evtq::qrct {

inline QRCTCalibration fit_gpd(const std::vector<float>& sorted_ratios) {
    size_t m = sorted_ratios.size();

    double sum = 0.0;
    for (size_t i = 0; i < m; ++i) sum += sorted_ratios[i];
    float mean = static_cast<float>(sum / static_cast<double>(m));
    auto it = std::lower_bound(sorted_ratios.begin(), sorted_ratios.end(), mean);
    size_t exc_start = static_cast<size_t>(std::distance(sorted_ratios.begin(), it));
    float u = sorted_ratios[exc_start];
    size_t n_exc = m - exc_start;

    double b0 = 0.0, b1 = 0.0;
    double nm1 = static_cast<double>(n_exc - 1);
    for (size_t i = 0; i < n_exc; ++i) {
        double x = static_cast<double>(sorted_ratios[exc_start + i] - u);
        b0 += x;
        b1 += x * static_cast<double>(i) / nm1;
    }
    b0 /= static_cast<double>(n_exc);
    b1 /= static_cast<double>(n_exc);

    double d = b0 - 2.0 * b1;
    float xi = (d > 0.0) ? std::max(-1.0f, static_cast<float>(2.0 - b0 / d)) : -1.0f;
    float sigma = static_cast<float>(b0) * (1.0f - xi);

    float p_u = static_cast<float>(n_exc) / static_cast<float>(m);
    return {xi, sigma, u, p_u, 0.0f};
}

template <size_t D>
QRCTCalibration calibrate(const RaBitQGraph<D>& graph, const NbitRaBitQEncoder<D>& encoder, size_t k, float target_recall, size_t num_probes) {
    size_t n = graph.size();

    std::mt19937 rng(static_cast<uint32_t>(n));
    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);
    indices.resize(num_probes);

    // Encode probe queries
    std::vector<AlignedVector<float>> padded_queries(num_probes);
    std::vector<RaBitQQuery<D>> encoded_queries(num_probes);
    for (size_t si = 0; si < num_probes; ++si) {
        padded_queries[si].resize(D);
        const float* qvec = graph.get_vector(static_cast<NodeId>(indices[si]));
        std::copy_n(qvec, graph.dim_, padded_queries[si].data());
        std::fill_n(padded_queries[si].data() + graph.dim_, D - graph.dim_, 0.0f);
        encoded_queries[si] = encoder.encode_query_raw(padded_queries[si].data());
    }

    // Collect estimation-error ratios from probe neighborhoods
    int num_threads = omp_get_max_threads();
    std::vector<std::vector<float>> per_thread(num_threads);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto& local = per_thread[tid];

        estimator::NeighborEstimates<D> estimates;

        #pragma omp for schedule(guided)
        for (size_t si = 0; si < num_probes; ++si) {
            NodeId qid = static_cast<NodeId>(indices[si]);
            const float* qvec = padded_queries[si].data();
            float qnsq = dot_product_simd<D>(qvec, qvec);

            const auto& probe_nb = graph.get_neighbors(qid);
            size_t num_pivots = probe_nb.size();

            for (size_t pi = 0; pi < num_pivots; ++pi) {
                NodeId pid = probe_nb.neighbor_ids[pi];
                float dqp = graph.query_distance(qvec, qnsq, pid);

                const auto& pnb = graph.get_neighbors(pid);
                size_t nn = pnb.size();
                estimator::estimate_all_neighbors<D>(encoded_queries[si], pnb, dqp, estimates);

                for (size_t i = 0; i < nn; ++i) {
                    NodeId nid = pnb.neighbor_ids[i];
                    if (nid == qid) continue;
                    float exact = graph.query_distance(qvec, qnsq, nid);
                    local.push_back(estimates.est_distances[i] / exact);
                }
            }
        }
    }

    size_t total = 0;
    for (auto& v : per_thread) total += v.size();
    std::vector<float> ratios;
    ratios.reserve(total);
    for (auto& v : per_thread) ratios.insert(ratios.end(), v.begin(), v.end());
    std::sort(ratios.begin(), ratios.end());

    auto cal = fit_gpd(ratios);

    // Ground truth via exhaustive search (negative delta is unreachable by
    // non-negative R_B, forcing the beam search to drain completely)
    QRCTCalibration cal_open = cal;
    cal_open.delta = -1.0f;

    std::vector<std::vector<NodeId>> ground_truth(num_probes);

    #pragma omp parallel
    {
        thread_local TwoLevelVisitationTable visited(0);
        if (visited.capacity() < n) visited.resize(n);

        #pragma omp for schedule(guided)
        for (size_t si = 0; si < num_probes; ++si) {
            auto full = rabitq_search::search<D>(encoded_queries[si], padded_queries[si].data(), graph, k, visited, graph.entry_point(), cal_open);
            ground_truth[si].reserve(full.size());
            for (const auto& sr : full) ground_truth[si].push_back(sr.id);
        }
    }

    // Binary search for delta
    float lo = 0.0f;
    float hi = static_cast<float>(k);
    float inv_k = 1.0f / static_cast<float>(k);

    while (hi - lo >= inv_k) {
        float mid = (lo + hi) * 0.5f;
        QRCTCalibration cal_test = cal;
        cal_test.delta = mid;

        float total_recall = 0.0f;

        #pragma omp parallel reduction(+:total_recall)
        {
            thread_local TwoLevelVisitationTable visited(0);
            if (visited.capacity() < n) visited.resize(n);

            #pragma omp for schedule(guided)
            for (size_t si = 0; si < num_probes; ++si) {
                auto term = rabitq_search::search<D>(encoded_queries[si], padded_queries[si].data(), graph, k, visited, graph.entry_point(), cal_test);

                size_t hits = 0;
                for (const auto& sr : term) {
                    for (NodeId fid : ground_truth[si]) {
                        if (sr.id == fid) { ++hits; break; }
                    }
                }
                total_recall += static_cast<float>(hits) / static_cast<float>(ground_truth[si].size());
            }
        }

        float mean_recall = total_recall / static_cast<float>(num_probes);
        if (mean_recall >= target_recall) {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    cal.delta = lo;
    return cal;
}

}
