#pragma once

#include "core.hpp"
#include "estimator.hpp"
#include "graph.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include <omp.h>

namespace evtq {

struct CalibrationSnapshot {
    float gamma;
    float dot_slack;
};

namespace calibration {

template <size_t D, typename EncType>
std::vector<float> collect_ratios(const RaBitQGraph<D>& graph, const EncType& encoder, size_t num_probes) {
    size_t n = graph.size();

    std::mt19937 rng(static_cast<uint32_t>(n ^ num_probes));
    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);
    indices.resize(num_probes);

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
            const float* qvec = graph.get_vector(qid);
            float qnsq = dot_product_simd<D>(qvec, qvec);

            RaBitQQuery<D> query = encoder.encode_query_raw(qvec);

            const auto& probe_nb = graph.get_neighbors(qid);
            size_t num_pivots = probe_nb.size();

            for (size_t pi = 0; pi < num_pivots; ++pi) {
                NodeId pid = probe_nb.neighbor_ids[pi];
                float dqp = graph.query_distance(qvec, qnsq, pid);

                const auto& pnb = graph.get_neighbors(pid);
                size_t nn = pnb.size();
                estimator::estimate_all_neighbors<D>(query, pnb, dqp, estimates);

                for (size_t i = 0; i < nn; ++i) {
                    NodeId nid = pnb.neighbor_ids[i];
                    if (nid == qid) continue;
                    float exact = graph.query_distance(qvec, qnsq, nid);
                    if (estimates.est_distances[i] > 0.0f && exact > 0.0f)
                        local.push_back(estimates.est_distances[i] / exact);
                }
            }
        }
    }

    size_t total = 0;
    for (auto& v : per_thread) total += v.size();
    std::vector<float> all;
    all.reserve(total);
    for (auto& v : per_thread) all.insert(all.end(), v.begin(), v.end());
    return all;
}

template <size_t D, typename EncType>
CalibrationSnapshot calibrate(const RaBitQGraph<D>& graph, const EncType& encoder, float dot_slack) {
    size_t n = graph.size();

    auto ratios = collect_ratios<D>(
        graph, encoder,
        static_cast<size_t>(std::ceil(std::sqrt(static_cast<double>(n)))));

    std::sort(ratios.begin(), ratios.end());
    size_t m = ratios.size();
    size_t half = m / 2;
    float threshold = ratios[half];

    size_t n_exc = m - half;
    double b0 = 0.0, b1 = 0.0;
    double nm1 = static_cast<double>(n_exc - 1);
    for (size_t i = 0; i < n_exc; ++i) {
        double x = static_cast<double>(ratios[half + i] - threshold);
        b0 += x;
        b1 += x * static_cast<double>(i) / nm1;
    }
    b0 /= static_cast<double>(n_exc);
    b1 /= static_cast<double>(n_exc);
    double d = b0 - 2.0 * b1;
    float xi = static_cast<float>(2.0 - b0 / d);
    float sigma = static_cast<float>(2.0 * b0 * b1 / d);

    float gamma;
    if (xi == 0.0f) {
        gamma = threshold + sigma * std::log(static_cast<float>(m) / 2.0f);
    } else {
        gamma = threshold
            + sigma / xi * (std::pow(static_cast<float>(m) / 2.0f, xi) - 1.0f);
    }

    return {gamma, dot_slack};
}

}
}