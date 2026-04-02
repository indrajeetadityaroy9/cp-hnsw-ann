#pragma once

#include "core.hpp"
#include <climits>
#include <limits>
#include <cmath>
#include <algorithm>

#include <immintrin.h>

namespace evtq {

template <size_t D>
struct FastScanCodeBlock {
    static constexpr size_t NUM_SUB_SEGMENTS = num_sub_segments<D>;
    static constexpr size_t NUM_SUB_PAIRS = (NUM_SUB_SEGMENTS + 1) / 2;

    alignas(SIMD_ALIGNMENT) uint8_t packed[NUM_SUB_PAIRS][32];

    void store(size_t idx, const BinaryCodeStorage<D>& code) {
        for (size_t sp = 0; sp < NUM_SUB_PAIRS; ++sp) {
            uint8_t lo = extract_sub_segment(code, 2 * sp);
            uint8_t hi = (2 * sp + 1 < NUM_SUB_SEGMENTS) ? extract_sub_segment(code, 2 * sp + 1) : 0;
            packed[sp][idx] = static_cast<uint8_t>((hi << 4) | lo);
        }
    }

    static uint8_t extract_sub_segment(const BinaryCodeStorage<D>& code, size_t seg_idx) {
        size_t bit_start = seg_idx * 4;
        uint8_t result = 0;
        for (size_t b = 0; b < 4 && (bit_start + b) < D; ++b) {
            if (code.get_bit(bit_start + b)) result |= (1 << b);
        }
        return result;
    }
};

template <size_t D>
struct NbitFastScanCodeBlock {
    FastScanCodeBlock<D> planes[BIT_WIDTH];

    void store(size_t idx, const NbitCodeStorage<D>& code) {
        for (size_t b = 0; b < BIT_WIDTH; ++b) {
            BinaryCodeStorage<D> plane_binary;
            std::memcpy(plane_binary.signs, code.planes[b],
                        BinaryCodeStorage<D>::NUM_WORDS * sizeof(uint64_t));
            planes[b].store(idx, plane_binary);
        }
    }
};

template <size_t D>
struct NbitFastScanNeighborBlock {
    static constexpr size_t NUM_BATCHES = GRAPH_DEGREE / (256 / CHAR_BIT);

    NbitFastScanCodeBlock<D> code_blocks[NUM_BATCHES];
    alignas(SIMD_ALIGNMENT) float centered_norm[GRAPH_DEGREE];
    alignas(SIMD_ALIGNMENT) float code_ip[GRAPH_DEGREE];
    alignas(SIMD_ALIGNMENT) float code_parent_ip[GRAPH_DEGREE];
    alignas(SIMD_ALIGNMENT) uint16_t popcounts[GRAPH_DEGREE];
    alignas(SIMD_ALIGNMENT) uint16_t msb2_popcounts[GRAPH_DEGREE];
    alignas(SIMD_ALIGNMENT) uint16_t weighted_popcounts[GRAPH_DEGREE];
    alignas(SIMD_ALIGNMENT) uint32_t neighbor_ids[GRAPH_DEGREE];
    uint32_t count;

    NbitFastScanNeighborBlock() : count(0) {
        std::memset(neighbor_ids, 0xFF, sizeof(neighbor_ids));
    }

    void set_neighbor(size_t slot, uint32_t id, const NbitCodeStorage<D>& code, const VertexAuxData& aux_data) {
        neighbor_ids[slot] = id;
        code_blocks[slot / 32].store(slot % 32, code);
        centered_norm[slot] = aux_data.centered_norm;
        code_ip[slot] = aux_data.code_ip;
        code_parent_ip[slot] = aux_data.code_parent_ip;
        popcounts[slot] = static_cast<uint16_t>(code.msb_popcount());
        msb2_popcounts[slot] = static_cast<uint16_t>(code.msb2_popcount());
        weighted_popcounts[slot] = static_cast<uint16_t>(code.weighted_popcount());
        if (slot >= count) count = static_cast<uint32_t>(slot + 1);
    }

    size_t size() const { return count; }
};

namespace fastscan {

constexpr size_t BATCH_SIZE = 256 / CHAR_BIT;

inline __m256 rcp_nr(__m256 d) {
    __m256 x = _mm256_rcp_ps(d);
    return _mm256_mul_ps(x, _mm256_fnmadd_ps(d, x, _mm256_set1_ps(2.0f)));
}

#ifdef __AVX512F__
inline __m512 rcp_nr(__m512 d) {
    __m512 x = _mm512_rcp14_ps(d);
    return _mm512_mul_ps(x, _mm512_fnmadd_ps(d, x, _mm512_set1_ps(2.0f)));
}
#endif

template <size_t D>
inline void compute_inner_products(const uint8_t lut[][16], const FastScanCodeBlock<D>& block, uint32_t* __restrict__ out) {
    constexpr size_t NUM_SUB_PAIRS = FastScanCodeBlock<D>::NUM_SUB_PAIRS;
    constexpr size_t NUM_SUB_SEGMENTS = FastScanCodeBlock<D>::NUM_SUB_SEGMENTS;
    constexpr size_t FLUSH = std::numeric_limits<uint8_t>::max() / (2 * BIT_WIDTH * ((1 << BIT_WIDTH) - 1));

    const __m256i low_mask = _mm256_set1_epi8(0x0F);
    __m256i acc_lo = _mm256_setzero_si256();
    __m256i acc_hi = _mm256_setzero_si256();
    __m256i tmp_acc = _mm256_setzero_si256();
    size_t pairs_since_flush = 0;

    for (size_t sp = 0; sp < NUM_SUB_PAIRS; ++sp) {
        __m256i codes = _mm256_load_si256(reinterpret_cast<const __m256i*>(block.packed[sp]));

        __m256i lo_nibbles = _mm256_and_si256(codes, low_mask);
        __m256i hi_nibbles = _mm256_and_si256(_mm256_srli_epi16(codes, 4), low_mask);

        __m256i lut_lo = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(lut[2 * sp])));
        tmp_acc = _mm256_add_epi8(tmp_acc, _mm256_shuffle_epi8(lut_lo, lo_nibbles));

        if (2 * sp + 1 < NUM_SUB_SEGMENTS) {
            __m256i lut_hi = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(lut[2 * sp + 1])));
            tmp_acc = _mm256_add_epi8(tmp_acc, _mm256_shuffle_epi8(lut_hi, hi_nibbles));
        }

        if (++pairs_since_flush >= FLUSH || sp == NUM_SUB_PAIRS - 1) {
            acc_lo = _mm256_add_epi16(acc_lo, _mm256_cvtepu8_epi16(_mm256_castsi256_si128(tmp_acc)));
            acc_hi = _mm256_add_epi16(acc_hi, _mm256_cvtepu8_epi16(_mm256_extracti128_si256(tmp_acc, 1)));
            tmp_acc = _mm256_setzero_si256();
            pairs_since_flush = 0;
        }
    }

    _mm256_store_si256(reinterpret_cast<__m256i*>(out), _mm256_cvtepu16_epi32(_mm256_castsi256_si128(acc_lo)));
    _mm256_store_si256(reinterpret_cast<__m256i*>(out + 8), _mm256_cvtepu16_epi32(_mm256_extracti128_si256(acc_lo, 1)));
    _mm256_store_si256(reinterpret_cast<__m256i*>(out + 16), _mm256_cvtepu16_epi32(_mm256_castsi256_si128(acc_hi)));
    _mm256_store_si256(reinterpret_cast<__m256i*>(out + 24), _mm256_cvtepu16_epi32(_mm256_extracti128_si256(acc_hi, 1)));
}

template <size_t D>
inline void compute_nbit_inner_products(const uint8_t lut[][16], const NbitFastScanCodeBlock<D>& block, uint32_t* __restrict__ out_nbit, uint32_t* __restrict__ out_msb) {
    std::memset(out_nbit, 0, 32 * sizeof(uint32_t));
    alignas(SIMD_ALIGNMENT) uint32_t plane_sums[32];

    for (size_t b = 0; b < BIT_WIDTH; ++b) {
        compute_inner_products<D>(lut, block.planes[b], plane_sums);
        if (b == 0) std::memcpy(out_msb, plane_sums, 32 * sizeof(uint32_t));
        __m256i vweight = _mm256_set1_epi32(1u << (BIT_WIDTH - 1 - b));
        for (size_t i = 0; i < 32; i += 8) {
            __m256i vout = _mm256_load_si256(reinterpret_cast<const __m256i*>(out_nbit + i));
            __m256i vplane = _mm256_load_si256(reinterpret_cast<const __m256i*>(plane_sums + i));
            _mm256_store_si256(reinterpret_cast<__m256i*>(out_nbit + i), _mm256_add_epi32(vout, _mm256_mullo_epi32(vweight, vplane)));
        }
    }
}

template <size_t D>
inline void convert_nbit_to_distances_with_bounds(const RaBitQQuery<D>& query, const uint32_t* nbit_fastscan_sums, const uint32_t* msb_fastscan_sums, const float* centered_norm_arr, const float* code_ip_arr, const float* code_parent_ip_arr, const uint16_t* msb_popcounts, const uint16_t* weighted_popcounts, size_t count, float* __restrict__ out_dist, float* __restrict__ out_lower, float dist_qp_sq) {
    constexpr float inv_K = 1.0f / static_cast<float>((1u << BIT_WIDTH) - 1);

    const float A_nbit = query.coeff_fastscan * inv_K;
    const float B_nbit = query.coeff_popcount * inv_K;
    const float C = query.coeff_constant;
    const float dot_slack = query.dot_slack;
    const float sqrt_dqp = std::sqrt(dist_qp_sq);

    size_t i = 0;

#ifdef __AVX512F__
    const __m512 vA_nbit_w = _mm512_set1_ps(A_nbit);
    const __m512 vB_nbit_w = _mm512_set1_ps(B_nbit);
    const __m512 vA_msb_w = _mm512_set1_ps(query.coeff_fastscan);
    const __m512 vB_msb_w = _mm512_set1_ps(query.coeff_popcount);
    const __m512 vC_w = _mm512_set1_ps(C);
    const __m512 vdot_slack_w = _mm512_set1_ps(dot_slack);
    const __m512 vsqrt_dqp_w = _mm512_set1_ps(sqrt_dqp);
    const __m512 vdist_qp_sq_w = _mm512_set1_ps(dist_qp_sq);
    const __m512 vone_w = _mm512_set1_ps(1.0f);
    const __m512 vneg_one_w = _mm512_set1_ps(-1.0f);
    const __m512 vrcp_sqrt_dqp_w = rcp_nr(vsqrt_dqp_w);

    for (; i + 16 <= count; i += 16) {
        __m512 nbit_fs = _mm512_cvtepi32_ps(_mm512_loadu_si512(nbit_fastscan_sums + i));
        __m512 vwpc = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(_mm256_load_si256(reinterpret_cast<const __m256i*>(weighted_popcounts + i))));
        __m512 ip_approx_nbit = _mm512_fmadd_ps(vA_nbit_w, nbit_fs, _mm512_fmadd_ps(vB_nbit_w, vwpc, vC_w));

        __m512 code_ip = _mm512_load_ps(code_ip_arr + i);
        __m512 code_parent_ip = _mm512_load_ps(code_parent_ip_arr + i);
        __m512 centered_norm = _mm512_load_ps(centered_norm_arr + i);
        __m512 rcp_code_ip = rcp_nr(code_ip);

        __m512 ip_est_nbit = _mm512_mul_ps(_mm512_sub_ps(ip_approx_nbit, code_parent_ip), rcp_code_ip);

        __m512 two_centered_norm = _mm512_add_ps(centered_norm, centered_norm);
        __m512 centered_norm_sq_plus_dqp = _mm512_fmadd_ps(centered_norm, centered_norm, vdist_qp_sq_w);

        _mm512_store_ps(out_dist + i, _mm512_fnmadd_ps(two_centered_norm, ip_est_nbit, centered_norm_sq_plus_dqp));

        __m512 msb_fs = _mm512_cvtepi32_ps(_mm512_loadu_si512(msb_fastscan_sums + i));
        __m512 vmpc = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(_mm256_load_si256(reinterpret_cast<const __m256i*>(msb_popcounts + i))));
        __m512 ip_est_msb = _mm512_mul_ps(_mm512_sub_ps(_mm512_fmadd_ps(vA_msb_w, msb_fs, _mm512_fmadd_ps(vB_msb_w, vmpc, vC_w)), code_parent_ip), rcp_code_ip);
        __m512 cos_upper = _mm512_min_ps(_mm512_max_ps(_mm512_mul_ps(_mm512_add_ps(ip_est_msb, vdot_slack_w), vrcp_sqrt_dqp_w), vneg_one_w), vone_w);
        _mm512_store_ps(out_lower + i, _mm512_fnmadd_ps(_mm512_mul_ps(two_centered_norm, vsqrt_dqp_w), cos_upper, centered_norm_sq_plus_dqp));
    }
#endif

    const __m256 vA_nbit = _mm256_set1_ps(A_nbit);
    const __m256 vB_nbit = _mm256_set1_ps(B_nbit);
    const __m256 vA_msb = _mm256_set1_ps(query.coeff_fastscan);
    const __m256 vB_msb = _mm256_set1_ps(query.coeff_popcount);
    const __m256 vC = _mm256_set1_ps(C);
    const __m256 vdot_slack = _mm256_set1_ps(dot_slack);
    const __m256 vsqrt_dqp = _mm256_set1_ps(sqrt_dqp);
    const __m256 vdist_qp_sq = _mm256_set1_ps(dist_qp_sq);
    const __m256 vone = _mm256_set1_ps(1.0f);
    const __m256 vneg_one = _mm256_set1_ps(-1.0f);
    const __m256 vrcp_sqrt_dqp = rcp_nr(vsqrt_dqp);

    for (; i + 8 <= count; i += 8) {
        __m256 nbit_fs = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(nbit_fastscan_sums + i)));
        __m256 vwpc = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm_load_si128(reinterpret_cast<const __m128i*>(weighted_popcounts + i))));
        __m256 ip_approx_nbit = _mm256_fmadd_ps(vA_nbit, nbit_fs, _mm256_fmadd_ps(vB_nbit, vwpc, vC));

        __m256 code_ip = _mm256_load_ps(code_ip_arr + i);
        __m256 code_parent_ip = _mm256_load_ps(code_parent_ip_arr + i);
        __m256 centered_norm = _mm256_load_ps(centered_norm_arr + i);
        __m256 rcp_code_ip = rcp_nr(code_ip);

        __m256 ip_est_nbit = _mm256_mul_ps(_mm256_sub_ps(ip_approx_nbit, code_parent_ip), rcp_code_ip);

        __m256 two_centered_norm = _mm256_add_ps(centered_norm, centered_norm);
        __m256 centered_norm_sq_plus_dqp = _mm256_fmadd_ps(centered_norm, centered_norm, vdist_qp_sq);

        _mm256_store_ps(out_dist + i, _mm256_fnmadd_ps(two_centered_norm, ip_est_nbit, centered_norm_sq_plus_dqp));

        __m256 msb_fs = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(msb_fastscan_sums + i)));
        __m256 vmpc = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm_load_si128(reinterpret_cast<const __m128i*>(msb_popcounts + i))));
        __m256 ip_est_msb = _mm256_mul_ps(_mm256_sub_ps(_mm256_fmadd_ps(vA_msb, msb_fs, _mm256_fmadd_ps(vB_msb, vmpc, vC)), code_parent_ip), rcp_code_ip);
        __m256 cos_upper = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(_mm256_add_ps(ip_est_msb, vdot_slack), vrcp_sqrt_dqp), vneg_one), vone);
        _mm256_store_ps(out_lower + i, _mm256_fnmadd_ps(_mm256_mul_ps(two_centered_norm, vsqrt_dqp), cos_upper, centered_norm_sq_plus_dqp));
    }

    for (; i < count; ++i) {
        float rcp_code_ip = 1.0f / code_ip_arr[i];
        float centered_norm = centered_norm_arr[i];
        float centered_norm_sq_dqp = centered_norm * centered_norm + dist_qp_sq;

        float ip_est_nbit = (A_nbit * static_cast<float>(nbit_fastscan_sums[i]) + B_nbit * static_cast<float>(weighted_popcounts[i]) + C - code_parent_ip_arr[i]) * rcp_code_ip;
        out_dist[i] = centered_norm_sq_dqp - 2.0f * centered_norm * ip_est_nbit;

        float ip_est_msb = (query.coeff_fastscan * static_cast<float>(msb_fastscan_sums[i]) + query.coeff_popcount * static_cast<float>(msb_popcounts[i]) + C - code_parent_ip_arr[i]) * rcp_code_ip;
        float cos_upper = std::clamp((ip_est_msb + dot_slack) / sqrt_dqp, -1.0f, 1.0f);
        out_lower[i] = centered_norm_sq_dqp - 2.0f * centered_norm * sqrt_dqp * cos_upper;
    }
}

template <size_t D>
inline void compute_msb_only_inner_products(const uint8_t lut[][16], const NbitFastScanCodeBlock<D>& block, uint32_t* __restrict__ out_msb) {
    compute_inner_products<D>(lut, block.planes[0], out_msb);

    {
        alignas(SIMD_ALIGNMENT) uint32_t plane1_sums[32];
        compute_inner_products<D>(lut, block.planes[1], plane1_sums);
        for (size_t i = 0; i < 32; i += 8) {
            __m256i vmsb = _mm256_load_si256(reinterpret_cast<const __m256i*>(out_msb + i));
            __m256i vp1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(plane1_sums + i));
            _mm256_store_si256(reinterpret_cast<__m256i*>(out_msb + i), _mm256_add_epi32(_mm256_slli_epi32(vmsb, 1), vp1));
        }
    }
}

template <size_t D>
inline void convert_msb_to_lower_bounds(const RaBitQQuery<D>& query, const uint32_t* msb_fastscan_sums, const float* centered_norm_arr, const float* code_ip_arr, const float* code_parent_ip_arr, const uint16_t* msb_popcounts, size_t count, float* __restrict__ out_lower, float dist_qp_sq) {
    constexpr float INV_K_PARTIAL = 1.0f / static_cast<float>((1 << (BIT_WIDTH / 2)) - 1);

    const float A = query.coeff_fastscan * INV_K_PARTIAL;
    const float B = query.coeff_popcount * INV_K_PARTIAL;
    const float C = query.coeff_constant;
    const float dot_slack = query.dot_slack;
    const float sqrt_dqp = std::sqrt(dist_qp_sq);

    size_t i = 0;

#ifdef __AVX512F__
    const __m512 vA_w = _mm512_set1_ps(A);
    const __m512 vB_w = _mm512_set1_ps(B);
    const __m512 vC_w = _mm512_set1_ps(C);
    const __m512 vdot_slack_w = _mm512_set1_ps(dot_slack);
    const __m512 vsqrt_dqp_w = _mm512_set1_ps(sqrt_dqp);
    const __m512 vdist_qp_sq_w = _mm512_set1_ps(dist_qp_sq);
    const __m512 vone_w = _mm512_set1_ps(1.0f);
    const __m512 vneg_one_w = _mm512_set1_ps(-1.0f);
    const __m512 vrcp_sqrt_dqp_w = rcp_nr(vsqrt_dqp_w);

    for (; i + 16 <= count; i += 16) {
        __m512 fs = _mm512_cvtepi32_ps(_mm512_loadu_si512(msb_fastscan_sums + i));
        __m512 vpc = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(_mm256_load_si256(reinterpret_cast<const __m256i*>(msb_popcounts + i))));
        __m512 ip_approx = _mm512_fmadd_ps(vA_w, fs, _mm512_fmadd_ps(vB_w, vpc, vC_w));

        __m512 code_ip = _mm512_load_ps(code_ip_arr + i);
        __m512 code_parent_ip = _mm512_load_ps(code_parent_ip_arr + i);
        __m512 centered_norm = _mm512_load_ps(centered_norm_arr + i);

        __m512 ip_est = _mm512_mul_ps(_mm512_sub_ps(ip_approx, code_parent_ip), rcp_nr(code_ip));

        __m512 cos_upper = _mm512_min_ps(_mm512_max_ps(_mm512_mul_ps(_mm512_add_ps(ip_est, vdot_slack_w), vrcp_sqrt_dqp_w), vneg_one_w), vone_w);
        __m512 two_centered_norm = _mm512_add_ps(centered_norm, centered_norm);
        _mm512_store_ps(out_lower + i, _mm512_fnmadd_ps(_mm512_mul_ps(two_centered_norm, vsqrt_dqp_w), cos_upper, _mm512_fmadd_ps(centered_norm, centered_norm, vdist_qp_sq_w)));
    }
#endif

    const __m256 vA = _mm256_set1_ps(A);
    const __m256 vB = _mm256_set1_ps(B);
    const __m256 vC = _mm256_set1_ps(C);
    const __m256 vdot_slack = _mm256_set1_ps(dot_slack);
    const __m256 vsqrt_dqp = _mm256_set1_ps(sqrt_dqp);
    const __m256 vdist_qp_sq = _mm256_set1_ps(dist_qp_sq);
    const __m256 vone = _mm256_set1_ps(1.0f);
    const __m256 vneg_one = _mm256_set1_ps(-1.0f);
    const __m256 vrcp_sqrt_dqp = rcp_nr(vsqrt_dqp);

    for (; i + 8 <= count; i += 8) {
        __m256 fs = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(msb_fastscan_sums + i)));
        __m256 vpc = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm_load_si128(reinterpret_cast<const __m128i*>(msb_popcounts + i))));
        __m256 ip_approx = _mm256_fmadd_ps(vA, fs, _mm256_fmadd_ps(vB, vpc, vC));

        __m256 code_ip = _mm256_load_ps(code_ip_arr + i);
        __m256 code_parent_ip = _mm256_load_ps(code_parent_ip_arr + i);
        __m256 centered_norm = _mm256_load_ps(centered_norm_arr + i);

        __m256 ip_est = _mm256_mul_ps(_mm256_sub_ps(ip_approx, code_parent_ip), rcp_nr(code_ip));

        __m256 cos_upper = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(_mm256_add_ps(ip_est, vdot_slack), vrcp_sqrt_dqp), vneg_one), vone);
        __m256 two_centered_norm = _mm256_add_ps(centered_norm, centered_norm);
        _mm256_store_ps(out_lower + i, _mm256_fnmadd_ps(_mm256_mul_ps(two_centered_norm, vsqrt_dqp), cos_upper, _mm256_fmadd_ps(centered_norm, centered_norm, vdist_qp_sq)));
    }

    for (; i < count; ++i) {
        float ip_approx = A * static_cast<float>(msb_fastscan_sums[i]) + B * static_cast<float>(msb_popcounts[i]) + C;
        float ip_est = (ip_approx - code_parent_ip_arr[i]) / code_ip_arr[i];
        float cos_upper = std::clamp((ip_est + dot_slack) / sqrt_dqp, -1.0f, 1.0f);
        out_lower[i] = centered_norm_arr[i] * centered_norm_arr[i] + dist_qp_sq - 2.0f * centered_norm_arr[i] * sqrt_dqp * cos_upper;
    }
}

}

}
