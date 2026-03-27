#pragma once

#include "../core/core.hpp"
#include <array>
#include <climits>
#include <cmath>
#include <algorithm>
#include <random>
#include <utility>
#include <vector>

#include <immintrin.h>
#include <omp.h>

namespace cphnsw {

// ============================================================
// Fast Hadamard Transform (formerly transform/fht.hpp)
// ============================================================

inline void fht(float* vec, size_t len) {
#ifdef __AVX512F__
    constexpr size_t SIMD_WIDTH = 16;

    // 4 in-register butterfly stages for 16-element blocks
    for (size_t i = 0; i < len; i += SIMD_WIDTH) {
        __m512 v = _mm512_loadu_ps(&vec[i]);

        // Stride 1: swap adjacent pairs
        __m512 v_swap1 = _mm512_permute_ps(v, 0b10110001);
        __m512 v_add1 = _mm512_add_ps(v, v_swap1);
        __m512 v_sub1 = _mm512_sub_ps(v, v_swap1);
        v = _mm512_mask_blend_ps(0xAAAA, v_add1, v_sub1);

        // Stride 2: swap pairs of 2
        __m512 v_swap2 = _mm512_permute_ps(v, 0b01001110);
        __m512 v_add2 = _mm512_add_ps(v, v_swap2);
        __m512 v_sub2 = _mm512_sub_ps(v, v_swap2);
        v = _mm512_mask_blend_ps(0xCCCC, v_add2, v_sub2);

        // Stride 4: swap lane0↔lane1 and lane2↔lane3 within each 256-bit half
        __m512 v_swap4 = _mm512_shuffle_f32x4(v, v, _MM_SHUFFLE(2, 3, 0, 1));
        __m512 v_add4 = _mm512_add_ps(v, v_swap4);
        __m512 v_sub4 = _mm512_sub_ps(v, v_swap4);
        v = _mm512_mask_blend_ps(0xF0F0, v_add4, v_sub4);

        // Stride 8: swap lower 256-bit half [lanes 0,1] ↔ upper [lanes 2,3]
        __m512 v_swap8 = _mm512_shuffle_f32x4(v, v, _MM_SHUFFLE(1, 0, 3, 2));
        __m512 v_add8 = _mm512_add_ps(v, v_swap8);
        __m512 v_sub8 = _mm512_sub_ps(v, v_swap8);
        v = _mm512_mask_blend_ps(0xFF00, v_add8, v_sub8);

        _mm512_storeu_ps(&vec[i], v);
    }

    for (size_t h = SIMD_WIDTH; h < len; h *= 2) {
        for (size_t i = 0; i < len; i += h * 2) {
            for (size_t j = i; j < i + h; j += SIMD_WIDTH) {
                __m512 x = _mm512_loadu_ps(&vec[j]);
                __m512 y = _mm512_loadu_ps(&vec[j + h]);
                _mm512_storeu_ps(&vec[j], _mm512_add_ps(x, y));
                _mm512_storeu_ps(&vec[j + h], _mm512_sub_ps(x, y));
            }
        }
    }
#else
    constexpr size_t SIMD_WIDTH = 8;

    for (size_t i = 0; i < len; i += SIMD_WIDTH) {
        __m256 v = _mm256_loadu_ps(&vec[i]);

        __m256 v_swap1 = _mm256_permute_ps(v, 0b10110001);
        __m256 v_add1 = _mm256_add_ps(v, v_swap1);
        __m256 v_sub1 = _mm256_sub_ps(v, v_swap1);
        v = _mm256_blend_ps(v_add1, v_sub1, 0b10101010);

        __m256 v_swap2 = _mm256_permute_ps(v, 0b01001110);
        __m256 v_add2 = _mm256_add_ps(v, v_swap2);
        __m256 v_sub2 = _mm256_sub_ps(v, v_swap2);
        v = _mm256_blend_ps(v_add2, v_sub2, 0b11001100);

        __m256 v_swap4 = _mm256_permute2f128_ps(v, v, 0x01);
        __m256 v_add4 = _mm256_add_ps(v, v_swap4);
        __m256 v_sub4 = _mm256_sub_ps(v, v_swap4);
        v = _mm256_blend_ps(v_add4, v_sub4, 0b11110000);

        _mm256_storeu_ps(&vec[i], v);
    }

    for (size_t h = SIMD_WIDTH; h < len; h *= 2) {
        for (size_t i = 0; i < len; i += h * 2) {
            for (size_t j = i; j < i + h; j += SIMD_WIDTH) {
                __m256 x = _mm256_loadu_ps(&vec[j]);
                __m256 y = _mm256_loadu_ps(&vec[j + h]);
                _mm256_storeu_ps(&vec[j], _mm256_add_ps(x, y));
                _mm256_storeu_ps(&vec[j + h], _mm256_sub_ps(x, y));
            }
        }
    }
#endif
}

// ============================================================
// Random Hadamard Rotation (formerly rotation.hpp)
// ============================================================

class RandomHadamardRotation {
public:
    static constexpr size_t NUM_LAYERS = 3;

    explicit RandomHadamardRotation(size_t dim, uint64_t seed = 42)
        : original_dim_(dim)
        , padded_dim_(next_power_of_two(dim)) {

        std::mt19937_64 rng(seed);
        std::uniform_int_distribution<int> coin(0, 1);

        for (size_t layer = 0; layer < NUM_LAYERS; ++layer) {
            signs_float_[layer].resize(padded_dim_);
            for (size_t i = 0; i < padded_dim_; ++i) {
                signs_float_[layer][i] = coin(rng) ? 1.0f : -1.0f;
            }
        }
    }

    void apply(float* x) const {
        apply_diagonal(x, signs_float_[0].data());
        fht(x, padded_dim_);

        apply_diagonal(x, signs_float_[1].data());
        fht(x, padded_dim_);

        apply_diagonal(x, signs_float_[2].data());
        fht(x, padded_dim_);
    }

    void apply_copy(const float* input, float* output) const {
        std::memcpy(output, input, original_dim_ * sizeof(float));
        std::memset(output + original_dim_, 0,
                    (padded_dim_ - original_dim_) * sizeof(float));

        apply(output);
    }

    size_t padded_dim() const { return padded_dim_; }

private:
    size_t original_dim_;
    size_t padded_dim_;
    std::array<AlignedVector<float>, NUM_LAYERS> signs_float_;

    void apply_diagonal(float* x, const float* signs_f) const {
#ifdef __AVX512F__
        for (size_t i = 0; i < padded_dim_; i += 16) {
            __m512 vx = _mm512_loadu_ps(x + i);
            __m512 vs = _mm512_load_ps(signs_f + i);
            _mm512_storeu_ps(x + i, _mm512_mul_ps(vx, vs));
        }
#else
        for (size_t i = 0; i < padded_dim_; i += 8) {
            __m256 vx = _mm256_loadu_ps(x + i);
            __m256 vs = _mm256_load_ps(signs_f + i);
            _mm256_storeu_ps(x + i, _mm256_mul_ps(vx, vs));
        }
#endif
    }
};

// ============================================================
// RaBitQ Encoder
// ============================================================

template <size_t D, size_t BitWidth>
class NbitRaBitQEncoder {
public:
    using CodeType = NbitRaBitQCode<D, BitWidth>;
    using QueryType = RaBitQQuery<D>;
    static constexpr size_t DIMS = D;
    static constexpr size_t BIT_WIDTH = BitWidth;
    static constexpr size_t NUM_SUB_SEGMENTS = num_sub_segments<D>;
    static constexpr int K_INT = (1 << BitWidth) - 1;
    static constexpr float K = static_cast<float>(K_INT);

    explicit NbitRaBitQEncoder(size_t dim, uint64_t rotation_seed = 42)
        : dim_(dim)
        , rotation_(dim, rotation_seed)
        , padded_dim_(rotation_.padded_dim())
        , centroid_(dim, 0.0f) {
        float d_float = static_cast<float>(D);
        norm_factor_ = 1.0f / (d_float * std::sqrt(d_float));
        inv_sqrt_d_ = 1.0f / std::sqrt(d_float);
    }

    void compute_centroid(const float* vecs, size_t num_vecs) {
        centroid_.assign(dim_, 0.0f);
        for (size_t i = 0; i < num_vecs; ++i) {
            const float* v = vecs + i * dim_;
            for (size_t j = 0; j < dim_; ++j) {
                centroid_[j] += v[j];
            }
        }
        float inv_n = 1.0f / static_cast<float>(num_vecs);
        for (size_t j = 0; j < dim_; ++j) {
            centroid_[j] *= inv_n;
        }
    }

    template <typename CT>
    void encode_batch(const float* vecs, size_t num_vecs, CT* codes) {
        compute_centroid(vecs, num_vecs);

        #pragma omp parallel
        {
            AlignedVector<float> buf(padded_dim_);
            std::vector<float> centered(dim_);

            #pragma omp for schedule(static)
            for (size_t i = 0; i < num_vecs; ++i) {
                codes[i] = encode_impl(
                    vecs + i * dim_, buf.data(), centered.data());
            }
        }
    }

    QueryType encode_query_raw(const float* vec) const {
        thread_local AlignedVector<float> buf(padded_dim_);
        thread_local AlignedVector<uint8_t> q_bar_u(padded_dim_);
        buf.resize(padded_dim_);
        q_bar_u.resize(padded_dim_);
        return encode_query_raw_impl(vec, buf.data(), q_bar_u.data());
    }

    void rotate_raw_vector(const float* vec, float* out) const {
        rotation_.apply_copy(vec, out);
#ifdef __AVX512F__
        __m512 vnf = _mm512_set1_ps(norm_factor_);
        for (size_t i = 0; i < padded_dim_; i += 16) {
            _mm512_storeu_ps(out + i, _mm512_mul_ps(_mm512_loadu_ps(out + i), vnf));
        }
#else
        __m256 vnf = _mm256_set1_ps(norm_factor_);
        for (size_t i = 0; i < padded_dim_; i += 8) {
            _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_loadu_ps(out + i), vnf));
        }
#endif
    }

    void build_lut(const float* buf, uint8_t* q_bar_u, QueryType& query) const {
        constexpr float LUT_LEVELS = static_cast<float>((1 << (CHAR_BIT / 2)) - 1);
        float vl = buf[0], vmax = buf[0];
        for (size_t i = 1; i < padded_dim_; ++i) {
            if (buf[i] < vl) vl = buf[i];
            if (buf[i] > vmax) vmax = buf[i];
        }

        float delta = (vmax - vl) / LUT_LEVELS;
        float inv_delta = 1.0f / delta;

        float sum_qu = 0.0f;
        for (size_t i = 0; i < padded_dim_; ++i) {
            float val = (buf[i] - vl) * inv_delta;
            int u = static_cast<int>(val + 0.5f);
            if (u < 0) u = 0;
            if (u > static_cast<int>(LUT_LEVELS)) u = static_cast<int>(LUT_LEVELS);
            q_bar_u[i] = static_cast<uint8_t>(u);
            sum_qu += static_cast<float>(u);
        }

        for (size_t j = 0; j < NUM_SUB_SEGMENTS; ++j) {
            for (uint8_t p = 0; p < 16; ++p) {
                uint8_t sum = 0;
                for (size_t b = 0; b < 4; ++b) {
                    size_t idx = j * 4 + b;
                    if (idx < D && (p & (1u << b))) {
                        sum += q_bar_u[idx];
                    }
                }
                query.lut[j][p] = sum;
            }
        }

        float Df = static_cast<float>(D);
        query.coeff_fastscan = 2.0f * delta * inv_sqrt_d_;
        query.coeff_popcount = 2.0f * vl * inv_sqrt_d_;
        query.coeff_constant = -(Df * vl + delta * sum_qu) * inv_sqrt_d_;
    }

    const std::vector<float>& get_centroid() const { return centroid_; }
    void set_centroid(std::vector<float> c) { centroid_ = std::move(c); }

    std::pair<NbitCodeStorage<D, BitWidth>, VertexAuxData> compute_neighbor_aux_nbit(
        const float* parent_vec, const float* neighbor_vec,
        const float* rotated_parent) const
    {
        NbitCodeStorage<D, BitWidth> result_code;
        VertexAuxData result_aux;
        result_code.clear();

        alignas(64) float diff[D];
        float nop_sq = 0.0f;
        for (size_t i = 0; i < dim_; ++i) {
            diff[i] = neighbor_vec[i] - parent_vec[i];
            nop_sq += diff[i] * diff[i];
        }
        for (size_t i = dim_; i < D; ++i) diff[i] = 0.0f;

        float nop = std::sqrt(nop_sq);
        result_aux.nop = nop;

        float inv_nop = 1.0f / nop;
        for (size_t i = 0; i < D; ++i) diff[i] *= inv_nop;

        alignas(64) float rotated[D];
        rotation_.apply_copy(diff, rotated);
#ifdef __AVX512F__
        { __m512 vnf = _mm512_set1_ps(norm_factor_);
        for (size_t i = 0; i < padded_dim_; i += 16)
            _mm512_storeu_ps(rotated + i, _mm512_mul_ps(_mm512_loadu_ps(rotated + i), vnf)); }
#else
        { __m256 vnf = _mm256_set1_ps(norm_factor_);
        for (size_t i = 0; i < padded_dim_; i += 8)
            _mm256_storeu_ps(rotated + i, _mm256_mul_ps(_mm256_loadu_ps(rotated + i), vnf)); }
#endif

        float ip_cp = 0.0f;
        result_aux.ip_qo = caq_quantize(rotated, result_code,
                                         rotated_parent, &ip_cp);
        result_aux.ip_cp = ip_cp;
        return {std::move(result_code), result_aux};
    }

    CodeType encode_impl(const float* vec, float* buf, float* centered_buf) const {
        CodeType code;
        code.clear();

        float norm_sq = 0.0f;
        for (size_t i = 0; i < dim_; ++i) {
            float v = vec[i] - centroid_[i];
            centered_buf[i] = v;
            norm_sq += v * v;
        }
        float norm = std::sqrt(norm_sq);
        code.nop = norm;

        float inv_norm = 1.0f / norm;
        for (size_t i = 0; i < dim_; ++i) centered_buf[i] *= inv_norm;

        rotation_.apply_copy(centered_buf, buf);
#ifdef __AVX512F__
        { __m512 vnf = _mm512_set1_ps(norm_factor_);
        for (size_t i = 0; i < padded_dim_; i += 16)
            _mm512_storeu_ps(buf + i, _mm512_mul_ps(_mm512_loadu_ps(buf + i), vnf)); }
#else
        { __m256 vnf = _mm256_set1_ps(norm_factor_);
        for (size_t i = 0; i < padded_dim_; i += 8)
            _mm256_storeu_ps(buf + i, _mm256_mul_ps(_mm256_loadu_ps(buf + i), vnf)); }
#endif

        code.ip_qo = caq_quantize(buf, code.codes);
        return code;
    }

private:
    size_t dim_;
    RandomHadamardRotation rotation_;
    size_t padded_dim_;
    float norm_factor_;
    float inv_sqrt_d_;
    std::vector<float> centroid_;

    QueryType encode_query_raw_impl(const float* vec, float* buf,
                                     uint8_t* q_bar_u) const {
        QueryType query;

        rotation_.apply_copy(vec, buf);
#ifdef __AVX512F__
        { __m512 vnf = _mm512_set1_ps(norm_factor_);
        for (size_t i = 0; i < padded_dim_; i += 16)
            _mm512_storeu_ps(buf + i, _mm512_mul_ps(_mm512_loadu_ps(buf + i), vnf)); }
#else
        { __m256 vnf = _mm256_set1_ps(norm_factor_);
        for (size_t i = 0; i < padded_dim_; i += 8)
            _mm256_storeu_ps(buf + i, _mm256_mul_ps(_mm256_loadu_ps(buf + i), vnf)); }
#endif

        build_lut(buf, q_bar_u, query);

        return query;
    }

    float caq_quantize(const float* rotated_buf,
                       NbitCodeStorage<D, BitWidth>& out_code,
                       const float* rotated_parent = nullptr,
                       float* out_ip_cp = nullptr) const {
        const size_t pd = padded_dim_;

        float buf_min = rotated_buf[0], buf_max = rotated_buf[0];
        for (size_t i = 1; i < pd; ++i) {
            if (rotated_buf[i] < buf_min) buf_min = rotated_buf[i];
            if (rotated_buf[i] > buf_max) buf_max = rotated_buf[i];
        }
        float delta = (buf_max - buf_min) / K;
        float inv_delta = 1.0f / delta;

        thread_local std::vector<int> codes_arr;
        codes_arr.resize(pd);

        float dot_co = 0.0f, norm_c_sq = 0.0f;
        for (size_t i = 0; i < pd; ++i) {
            float val = (rotated_buf[i] - buf_min) * inv_delta;
            int u = static_cast<int>(val + 0.5f);
            if (u < 0) u = 0;
            if (u > K_INT) u = K_INT;
            codes_arr[i] = u;
            float c = (2.0f * u - K) / K;
            dot_co += c * rotated_buf[i];
            norm_c_sq += c * c;
        }

        float prev_cos_sq = 0.0f;
        for (size_t iter = 0; ; ++iter) {
            bool changed = false;
            for (size_t i = 0; i < pd; ++i) {
                int old_u = codes_arr[i];
                float old_c = (2.0f * old_u - K) / K;
                float dot_without = dot_co - old_c * rotated_buf[i];
                float norm_without = norm_c_sq - old_c * old_c;
                int best_u = old_u;
                float best_dot = dot_co, best_norm = norm_c_sq;
                if constexpr (BitWidth >= 4) {
                    for (int delta : {-1, +1}) {
                        int u_try = old_u + delta;
                        if (u_try < 0 || u_try > K_INT) continue;
                        float c = (2.0f * u_try - K) / K;
                        float new_dot = dot_without + c * rotated_buf[i];
                        float new_norm = norm_without + c * c;
                        if (new_dot * new_dot * best_norm > best_dot * best_dot * new_norm) {
                            best_u = u_try;
                            best_dot = new_dot;
                            best_norm = new_norm;
                        }
                    }
                } else {
                    for (int u = 0; u <= K_INT; ++u) {
                        if (u == old_u) continue;
                        float c = (2.0f * u - K) / K;
                        float new_dot = dot_without + c * rotated_buf[i];
                        float new_norm = norm_without + c * c;
                        if (new_dot * new_dot * best_norm > best_dot * best_dot * new_norm) {
                            best_u = u;
                            best_dot = new_dot;
                            best_norm = new_norm;
                        }
                    }
                }
                if (best_u != old_u) {
                    float new_c = (2.0f * best_u - K) / K;
                    dot_co = dot_without + new_c * rotated_buf[i];
                    norm_c_sq = norm_without + new_c * new_c;
                    codes_arr[i] = best_u;
                    changed = true;
                }
            }
            if (!changed) break;
            float cos_sq = (norm_c_sq > 0.0f) ? (dot_co * dot_co / norm_c_sq) : 0.0f;
            if (iter > 0 && (cos_sq - prev_cos_sq) < 1.0f / (K * K)) break;
            prev_cos_sq = cos_sq;
        }

        float ip_qo = 0.0f;
        float ip_cp = 0.0f;
        for (size_t i = 0; i < pd; ++i) {
            out_code.set_value(i, static_cast<uint8_t>(codes_arr[i]));
            float c = (2.0f * codes_arr[i] - K) / K;
            ip_qo += c * rotated_buf[i];
            if (rotated_parent) ip_cp += c * rotated_parent[i];
        }
        if (out_ip_cp) *out_ip_cp = ip_cp * inv_sqrt_d_;
        return ip_qo * inv_sqrt_d_;
    }
};

}
