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
inline void estimate_neighbors(const RaBitQQuery<D>& query, const NbitFastScanNeighborBlock<D>& neighbors, float dist_qp_sq, float threshold, bool prune, NeighborEstimates<D>& out) {
    size_t n_neighbors = neighbors.size();
    size_t num_batches = (n_neighbors + fastscan::BATCH_SIZE - 1) / fastscan::BATCH_SIZE;

    for (size_t batch = 0; batch < num_batches; ++batch) {
        size_t start = batch * fastscan::BATCH_SIZE;
        if (start >= n_neighbors) break;
        size_t count = std::min(fastscan::BATCH_SIZE, n_neighbors - start);

        if (prune) {
            fastscan::compute_msb_only_inner_products<D>(query.lut, neighbors.code_blocks[batch], out.msb_sums + start);
            fastscan::convert_msb_to_lower_bounds<D>(query, out.msb_sums + start, neighbors.centered_norm + start, neighbors.code_ip + start, neighbors.code_parent_ip + start, neighbors.msb2_popcounts + start, count, out.lower_bounds + start, dist_qp_sq);

            bool any_survivor = false;
            for (size_t j = 0; j < count; ++j)
                if (out.lower_bounds[start + j] < threshold) { any_survivor = true; break; }

            if (!any_survivor) {
                for (size_t j = 0; j < count; ++j) out.est_distances[start + j] = std::numeric_limits<float>::max();
                continue;
            }
        }

        fastscan::compute_nbit_inner_products<D>(query.lut, neighbors.code_blocks[batch], out.fastscan_sums + start, out.msb_sums + start);
        fastscan::convert_nbit_to_distances_with_bounds<D>(query, out.fastscan_sums + start, out.msb_sums + start, neighbors.centered_norm + start, neighbors.code_ip + start, neighbors.code_parent_ip + start, neighbors.popcounts + start, neighbors.weighted_popcounts + start, count, out.est_distances + start, out.lower_bounds + start, dist_qp_sq);
    }
}

}
