#pragma once

#include "core.hpp"
#include "fastscan.hpp"

#include <algorithm>
#include <limits>

namespace evtq::estimator {

template <size_t D>
struct NeighborEstimates {
    alignas(SIMD_ALIGNMENT) uint32_t fastscan_sums[GRAPH_DEGREE];
    alignas(SIMD_ALIGNMENT) uint32_t msb_sums[GRAPH_DEGREE];
    alignas(SIMD_ALIGNMENT) float est_distances[GRAPH_DEGREE];
    alignas(SIMD_ALIGNMENT) float lower_bounds[GRAPH_DEGREE];
};

template <size_t D>
inline void estimate_all_neighbors(
    const RaBitQQuery<D>& query,
    const NbitFastScanNeighborBlock<D>& neighbors,
    float dist_qp_sq,
    NeighborEstimates<D>& out)
{
    constexpr size_t BATCH = fastscan::BATCH_SIZE;
    size_t n_neighbors = neighbors.size();
    size_t num_batches = (n_neighbors + BATCH - 1) / BATCH;

    for (size_t batch = 0; batch < num_batches; ++batch) {
        size_t batch_start = batch * BATCH;
        if (batch_start >= n_neighbors) break;
        size_t batch_count = std::min(BATCH, n_neighbors - batch_start);

        fastscan::compute_nbit_inner_products<D>(
            query.lut, neighbors.code_blocks[batch],
            out.fastscan_sums + batch_start, out.msb_sums + batch_start);
        fastscan::convert_nbit_to_distances_with_bounds<D>(
            query, out.fastscan_sums + batch_start,
            out.msb_sums + batch_start,
            neighbors.centered_norm + batch_start,
            neighbors.code_ip + batch_start,
            neighbors.code_parent_ip + batch_start,
            neighbors.popcounts + batch_start,
            neighbors.weighted_popcounts + batch_start,
            batch_count, out.est_distances + batch_start,
            out.lower_bounds + batch_start, dist_qp_sq);
    }
}

template <size_t D>
inline void estimate_neighbors_for_search(
    const RaBitQQuery<D>& query,
    const NbitFastScanNeighborBlock<D>& neighbors,
    float dist_qp_sq,
    float threshold,
    bool prune_full_estimates,
    NeighborEstimates<D>& out)
{
    constexpr size_t BATCH = fastscan::BATCH_SIZE;
    size_t n_neighbors = neighbors.size();
    size_t num_batches = (n_neighbors + BATCH - 1) / BATCH;

    for (size_t batch = 0; batch < num_batches; ++batch) {
        size_t batch_start = batch * BATCH;
        if (batch_start >= n_neighbors) break;
        size_t batch_count = std::min(BATCH, n_neighbors - batch_start);

        if (batch + 1 < num_batches && (batch + 1) * BATCH < n_neighbors) {
            prefetch_t<0>(reinterpret_cast<const char*>(&neighbors.code_blocks[batch + 1]));
        }

        fastscan::compute_msb_only_inner_products<D>(
            query.lut, neighbors.code_blocks[batch], out.msb_sums + batch_start);
        fastscan::convert_msb_to_lower_bounds<D>(
            query, out.msb_sums + batch_start,
            neighbors.centered_norm + batch_start,
            neighbors.code_ip + batch_start,
            neighbors.code_parent_ip + batch_start,
            neighbors.popcounts + batch_start,
            batch_count, out.lower_bounds + batch_start, dist_qp_sq);

        bool any_survivor = !prune_full_estimates;
        if (prune_full_estimates) {
            for (size_t j = 0; j < batch_count; ++j) {
                if (out.lower_bounds[batch_start + j] < threshold) {
                    any_survivor = true;
                    break;
                }
            }
        }

        if (!any_survivor) {
            for (size_t j = 0; j < batch_count; ++j) {
                out.est_distances[batch_start + j] = std::numeric_limits<float>::max();
            }
            continue;
        }

        fastscan::compute_nbit_inner_products<D>(
            query.lut, neighbors.code_blocks[batch],
            out.fastscan_sums + batch_start, out.msb_sums + batch_start);
        fastscan::convert_nbit_to_distances_with_bounds<D>(
            query, out.fastscan_sums + batch_start,
            out.msb_sums + batch_start,
            neighbors.centered_norm + batch_start,
            neighbors.code_ip + batch_start,
            neighbors.code_parent_ip + batch_start,
            neighbors.popcounts + batch_start,
            neighbors.weighted_popcounts + batch_start,
            batch_count, out.est_distances + batch_start,
            out.lower_bounds + batch_start, dist_qp_sq);
    }
}

}
