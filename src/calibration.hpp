#pragma once

#include "core.hpp"
#include "fastscan.hpp"
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

// §4.1 — Sample: collect estimation ratios from the 2-hop neighborhood of
// ⌈√n⌉ uniformly random probe nodes.
template <size_t D, size_t BitWidth, typename EncType>
std::vector<float> collect_ratios(
    const RaBitQGraph<D, BitWidth>& graph,
    const EncType& encoder,
    size_t num_probes)
{
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

        alignas(SIMD_ALIGNMENT) uint32_t fscan[GRAPH_DEGREE];
        alignas(SIMD_ALIGNMENT) uint32_t msb[GRAPH_DEGREE];
        alignas(SIMD_ALIGNMENT) float est[GRAPH_DEGREE];
        alignas(SIMD_ALIGNMENT) float lb[GRAPH_DEGREE];

        #pragma omp for schedule(guided)
        for (size_t si = 0; si < num_probes; ++si) {
            NodeId qid = static_cast<NodeId>(indices[si]);
            const float* qvec = graph.get_vector(qid);
            float qnsq = dot_product_simd<D>(qvec, qvec);

            RaBitQQuery<D> query = encoder.encode_query_raw(qvec);

            // Iterate all neighbors of the probe as pivots (2-hop: probe → pivot → pivot's neighbors)
            const auto& probe_nb = graph.get_neighbors(qid);
            size_t num_pivots = probe_nb.size();

            for (size_t pi = 0; pi < num_pivots; ++pi) {
                NodeId pid = probe_nb.neighbor_ids[pi];

                float dqp = qnsq + graph.get_norm_sq(pid)
                          - 2.0f * dot_product_simd<D>(qvec, graph.get_vector(pid));

                const auto& pnb = graph.get_neighbors(pid);
                size_t nn = pnb.size();

                constexpr size_t BATCH = fastscan::BATCH_SIZE;
                size_t nb = (GRAPH_DEGREE + BATCH - 1) / BATCH;

                for (size_t b = 0; b < nb; ++b) {
                    size_t bs = b * BATCH;
                    if (bs >= nn) break;
                    size_t bc = std::min(BATCH, nn - bs);

                    fastscan::compute_nbit_inner_products<D, BitWidth>(
                        query.lut, pnb.code_blocks[b], fscan + bs, msb + bs);
                    fastscan::convert_nbit_to_distances_with_bounds<D, BitWidth>(
                        query, fscan + bs, msb + bs,
                        pnb.centered_norm + bs, pnb.code_ip + bs, pnb.code_parent_ip + bs,
                        pnb.popcounts + bs, pnb.weighted_popcounts + bs,
                        bc, est + bs, lb + bs, dqp);
                }

                for (size_t i = 0; i < nn; ++i) {
                    NodeId nid = pnb.neighbor_ids[i];
                    if (nid == qid) continue;
                    float exact = qnsq + graph.get_norm_sq(nid)
                        - 2.0f * dot_product_simd<D>(qvec, graph.get_vector(nid));
                    if (est[i] > 0.0f && exact > 0.0f)
                        local.push_back(est[i] / exact);
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

// §4.1–4.4 — The calibration mechanism: sample, threshold, fit, extract.
template <size_t D, size_t BitWidth, typename EncType>
CalibrationSnapshot calibrate(
    const RaBitQGraph<D, BitWidth>& graph,
    const EncType& encoder,
    float dot_slack)
{
    size_t n = graph.size();

    // §4.1 — Sample: ⌈√n⌉ probes, 2-hop neighborhoods
    auto ratios = collect_ratios<D, BitWidth>(
        graph, encoder,
        static_cast<size_t>(std::ceil(std::sqrt(static_cast<double>(n)))));

    // §4.2 — Threshold: u = median(R)
    std::sort(ratios.begin(), ratios.end());
    size_t m = ratios.size();
    size_t half = m / 2;
    float threshold = ratios[half];

    // §4.3 — Fit: PWM estimation (Hosking & Wallis 1987) on above-median exceedances
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

    // §4.4 — Extract: GPD quantile at return period 1/m (Theorem 1)
    float gamma;
    if (xi == 0.0f) {
        // ξ = 0: γ = u + σ ln(m/2)
        gamma = threshold + sigma * std::log(static_cast<float>(m) / 2.0f);
    } else {
        // ξ ≠ 0: γ = u + (σ/ξ)[(m/2)^ξ − 1]
        gamma = threshold
            + sigma / xi * (std::pow(static_cast<float>(m) / 2.0f, xi) - 1.0f);
    }

    return {gamma, dot_slack};
}

} // namespace calibration
} // namespace evtq
