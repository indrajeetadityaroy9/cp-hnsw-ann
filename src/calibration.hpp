#pragma once

#include "core.hpp"
#include "encoder.hpp"
#include "estimator.hpp"
#include "graph.hpp"
#include <algorithm>
#include <numeric>
#include <random>
#include <vector>
#include <omp.h>

namespace evtq::rcgr {

inline GPDCalibration fit_gpd(const std::vector<float>& sorted_ratios) {
    size_t m = sorted_ratios.size();

    double sum = 0.0;
    for (size_t i = 0; i < m; ++i) sum += sorted_ratios[i];
    float mean = static_cast<float>(sum / static_cast<double>(m));
    auto it = std::lower_bound(sorted_ratios.begin(), sorted_ratios.end(), mean);
    size_t exc_start = static_cast<size_t>(std::distance(sorted_ratios.begin(), it));
    float u = sorted_ratios[exc_start];
    size_t n_exc = m - exc_start;

    double b0 = 0.0, b1 = 0.0, nm1 = static_cast<double>(n_exc - 1);
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
    return {xi, sigma, u, p_u};
}

template <size_t D>
GPDCalibration calibrate(const RaBitQGraph<D>& graph, const NbitRaBitQEncoder<D>& encoder, size_t num_probes) {
    size_t n = graph.size();
    std::mt19937 rng(static_cast<uint32_t>(n));
    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);
    indices.resize(num_probes);

    std::vector<AlignedVector<float>> padded_queries(num_probes);
    std::vector<RaBitQQuery<D>> encoded_queries(num_probes);
    for (size_t si = 0; si < num_probes; ++si) {
        padded_queries[si].resize(D);
        const float* qvec = graph.get_vector(static_cast<NodeId>(indices[si]));
        std::copy_n(qvec, graph.dim_, padded_queries[si].data());
        std::fill_n(padded_queries[si].data() + graph.dim_, D - graph.dim_, 0.0f);
        encoded_queries[si] = encoder.encode_query_raw(padded_queries[si].data());
    }

    int num_threads = omp_get_max_threads();
    std::vector<std::vector<float>> per_thread(num_threads);

    #pragma omp parallel
    {
        auto& local = per_thread[omp_get_thread_num()];
        estimator::NeighborEstimates<D> estimates;

        #pragma omp for schedule(guided)
        for (size_t si = 0; si < num_probes; ++si) {
            auto qid = static_cast<NodeId>(indices[si]);
            const float* qvec = padded_queries[si].data();
            float qnsq = dot_product_simd<D>(qvec, qvec);
            const auto& probe_nb = graph.get_neighbors(qid);

            for (size_t pi = 0; pi < probe_nb.size(); ++pi) {
                NodeId pid = probe_nb.neighbor_ids[pi];
                float dqp = graph.query_distance(qvec, qnsq, pid);
                const auto& pnb = graph.get_neighbors(pid);
                estimator::estimate_neighbors<D>(encoded_queries[si], pnb, dqp, 0.0f, false, estimates);

                for (size_t i = 0; i < pnb.size(); ++i) {
                    if (pnb.neighbor_ids[i] == qid) continue;
                    float exact = graph.query_distance(qvec, qnsq, pnb.neighbor_ids[i]);
                    local.push_back(estimates.est_distances[i] / exact);
                }
            }
        }
    }

    std::vector<float> ratios;
    size_t total = 0;
    for (auto& v : per_thread) total += v.size();
    ratios.reserve(total);
    for (auto& v : per_thread) ratios.insert(ratios.end(), v.begin(), v.end());
    std::sort(ratios.begin(), ratios.end());
    return fit_gpd(ratios);
}

}
