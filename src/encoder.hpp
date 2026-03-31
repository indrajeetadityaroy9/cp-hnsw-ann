#pragma once

#include "core.hpp"
#include <array>
#include <climits>
#include <cmath>
#include <algorithm>
#include <random>
#include <utility>
#include <vector>

#include <immintrin.h>
#include <omp.h>

namespace evtq {

inline void fht(float* vec, size_t len) {
#ifdef __AVX512F__
    constexpr size_t SIMD_WIDTH = 16;

    for (size_t i = 0; i < len; i += SIMD_WIDTH) {
        __m512 v = _mm512_loadu_ps(&vec[i]);

        __m512 v_swap1 = _mm512_permute_ps(v, 0b10110001);
        __m512 v_add1 = _mm512_add_ps(v, v_swap1);
        __m512 v_sub1 = _mm512_sub_ps(v, v_swap1);
        v = _mm512_mask_blend_ps(0xAAAA, v_add1, v_sub1);

        __m512 v_swap2 = _mm512_permute_ps(v, 0b01001110);
        __m512 v_add2 = _mm512_add_ps(v, v_swap2);
        __m512 v_sub2 = _mm512_sub_ps(v, v_swap2);
        v = _mm512_mask_blend_ps(0xCCCC, v_add2, v_sub2);

        __m512 v_swap4 = _mm512_shuffle_f32x4(v, v, _MM_SHUFFLE(2, 3, 0, 1));
        __m512 v_add4 = _mm512_add_ps(v, v_swap4);
        __m512 v_sub4 = _mm512_sub_ps(v, v_swap4);
        v = _mm512_mask_blend_ps(0xF0F0, v_add4, v_sub4);

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

struct RandomHadamardRotation {
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

template <size_t D>
struct NbitRaBitQEncoder {
    using CodeType = NbitRaBitQCode<D>;
    using QueryType = RaBitQQuery<D>;
    static constexpr size_t DIMS = D;
    static constexpr size_t NUM_SUB_SEGMENTS = num_sub_segments<D>;
    static constexpr int K_INT = (1 << BIT_WIDTH) - 1;
    static constexpr float K = static_cast<float>(K_INT);

    explicit NbitRaBitQEncoder(size_t dim, uint64_t rotation_seed = 42)
        : dim_(dim)
        , rotation_(dim, rotation_seed)
        , centroid_(dim, 0.0f) {
        float d_float = static_cast<float>(D);
        norm_factor_ = 1.0f / (d_float * std::sqrt(d_float));
        inv_sqrt_d_ = 1.0f / std::sqrt(d_float);
    }

    void encode_batch(const float* vecs, size_t num_vecs, CodeType* codes) {
        compute_centroid(vecs, num_vecs);

        #pragma omp parallel
        {
            AlignedVector<float> buf(D);
            AlignedVector<float> centered(D);

            #pragma omp for schedule(static)
            for (size_t i = 0; i < num_vecs; ++i) {
                codes[i] = encode_impl(
                    vecs + i * dim_, buf.data(), centered.data());
            }
        }
    }

    QueryType encode_query_raw(const float* vec) const {
        thread_local AlignedVector<float> buf(D);
        thread_local AlignedVector<uint8_t> quantized_query(D);

        QueryType query;
        rotate_and_normalize(vec, buf.data());
        build_lut(buf.data(), quantized_query.data(), query);
        return query;
    }

    void rotate_and_normalize(const float* input, float* output) const {
        rotation_.apply_copy(input, output);
#ifdef __AVX512F__
        __m512 vnf = _mm512_set1_ps(norm_factor_);
        for (size_t i = 0; i < D; i += 16)
            _mm512_storeu_ps(output + i, _mm512_mul_ps(_mm512_loadu_ps(output + i), vnf));
#else
        __m256 vnf = _mm256_set1_ps(norm_factor_);
        for (size_t i = 0; i < D; i += 8)
            _mm256_storeu_ps(output + i, _mm256_mul_ps(_mm256_loadu_ps(output + i), vnf));
#endif
    }

    const std::vector<float>& get_centroid() const { return centroid_; }
    void set_centroid(std::vector<float> c) { centroid_ = std::move(c); }

    std::pair<NbitCodeStorage<D>, VertexAuxData> compute_neighbor_aux_nbit(const float* parent_vec, const float* neighbor_vec, const float* rotated_parent) const {
        NbitCodeStorage<D> result_code;
        VertexAuxData result_aux;
        result_code.clear();

        alignas(SIMD_ALIGNMENT) float diff[D];
        result_aux.centered_norm =
            difference_and_normalize(parent_vec, neighbor_vec, diff);

        alignas(SIMD_ALIGNMENT) float rotated[D];
        rotate_and_normalize(diff, rotated);

        float code_parent_ip = 0.0f;
        result_aux.code_ip = caq_quantize(rotated, result_code,
                                         rotated_parent, &code_parent_ip);
        result_aux.code_parent_ip = code_parent_ip;
        return {std::move(result_code), result_aux};
    }

    float compute_dot_slack() const {
        return std::sqrt((static_cast<float>(M_PI) / 2.0f - 1.0f)
                         / static_cast<float>((1u << (2 * BIT_WIDTH)) * D));
    }

    size_t dim_;
    RandomHadamardRotation rotation_;
    float norm_factor_;
    float inv_sqrt_d_;
    std::vector<float> centroid_;

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

    void build_lut(const float* buf, uint8_t* quantized_query, QueryType& query) const {
        constexpr float LUT_LEVELS = static_cast<float>((1 << (CHAR_BIT / 2)) - 1);
        float lut_min = buf[0], lut_max = buf[0];
        for (size_t i = 1; i < D; ++i) {
            if (buf[i] < lut_min) lut_min = buf[i];
            if (buf[i] > lut_max) lut_max = buf[i];
        }

        float delta = (lut_max - lut_min) / LUT_LEVELS;
        float inv_delta = 1.0f / delta;

        float quantized_sum = 0.0f;
        for (size_t i = 0; i < D; ++i) {
            int u = static_cast<int>((buf[i] - lut_min) * inv_delta + 0.5f);
            quantized_query[i] = static_cast<uint8_t>(u);
            quantized_sum += static_cast<float>(u);
        }

        for (size_t j = 0; j < NUM_SUB_SEGMENTS; ++j) {
            for (uint8_t p = 0; p < 16; ++p) {
                uint8_t sum = 0;
                for (size_t b = 0; b < 4; ++b) {
                    size_t idx = j * 4 + b;
                    if (idx < D && (p & (1u << b))) {
                        sum += quantized_query[idx];
                    }
                }
                query.lut[j][p] = sum;
            }
        }

        float Df = static_cast<float>(D);
        query.coeff_fastscan = 2.0f * delta * inv_sqrt_d_;
        query.coeff_popcount = 2.0f * lut_min * inv_sqrt_d_;
        query.coeff_constant = -(Df * lut_min + delta * quantized_sum) * inv_sqrt_d_;
    }

    CodeType encode_impl(const float* vec, float* buf, float* centered_buf) const {
        CodeType code;
        code.clear();

        code.centered_norm = center_and_normalize(vec, centered_buf);
        rotate_and_normalize(centered_buf, buf);

        code.code_ip = caq_quantize(buf, code.codes);
        return code;
    }

    float center_and_normalize(const float* vec, float* out) const {
        float norm_sq = 0.0f;
        for (size_t i = 0; i < dim_; ++i) {
            float centered = vec[i] - centroid_[i];
            out[i] = centered;
            norm_sq += centered * centered;
        }
        for (size_t i = dim_; i < D; ++i) out[i] = 0.0f;

        float norm = std::sqrt(norm_sq);
        float inv_norm = 1.0f / norm;
        for (size_t i = 0; i < D; ++i) out[i] *= inv_norm;
        return norm;
    }

    float difference_and_normalize(const float* from, const float* to, float* out) const {
        float norm_sq = 0.0f;
        for (size_t i = 0; i < dim_; ++i) {
            float diff = to[i] - from[i];
            out[i] = diff;
            norm_sq += diff * diff;
        }
        for (size_t i = dim_; i < D; ++i) out[i] = 0.0f;

        float norm = std::sqrt(norm_sq);
        float inv_norm = 1.0f / norm;
        for (size_t i = 0; i < D; ++i) out[i] *= inv_norm;
        return norm;
    }

    float caq_quantize(const float* rotated_buf, NbitCodeStorage<D>& out_code, const float* rotated_parent = nullptr, float* out_code_parent_ip = nullptr) const {
        float buf_min = rotated_buf[0], buf_max = rotated_buf[0];
        for (size_t i = 1; i < D; ++i) {
            if (rotated_buf[i] < buf_min) buf_min = rotated_buf[i];
            if (rotated_buf[i] > buf_max) buf_max = rotated_buf[i];
        }
        float range = buf_max - buf_min;
        float inv_range = 1.0f / range;

        thread_local std::vector<int> levels;
        levels.resize(D);

        float code_vec_ip = 0.0f, code_norm_sq = 0.0f;
        for (size_t i = 0; i < D; ++i) {
            int u = static_cast<int>((rotated_buf[i] - buf_min) * inv_range * K + 0.5f);
            levels[i] = u;
            float coeff = (2.0f * u - K) / K;
            code_vec_ip += coeff * rotated_buf[i];
            code_norm_sq += coeff * coeff;
        }

        float prev_cosine_sq = 0.0f;
        for (size_t iter = 0; ; ++iter) {
            bool changed = false;
            for (size_t i = 0; i < D; ++i) {
                int old_level = levels[i];
                float old_coeff = (2.0f * old_level - K) / K;
                float ip_without_dim = code_vec_ip - old_coeff * rotated_buf[i];
                float norm_sq_without_dim = code_norm_sq - old_coeff * old_coeff;
                int best_level = old_level;
                float best_ip = code_vec_ip, best_norm_sq = code_norm_sq;
                for (int d : {-1, +1}) {
                    int try_level = old_level + d;
                    if (try_level < 0 || try_level > K_INT) continue;
                    float coeff = (2.0f * try_level - K) / K;
                    float cand_ip = ip_without_dim + coeff * rotated_buf[i];
                    float cand_norm_sq = norm_sq_without_dim + coeff * coeff;
                    if (cand_ip * cand_ip * best_norm_sq > best_ip * best_ip * cand_norm_sq) {
                        best_level = try_level;
                        best_ip = cand_ip;
                        best_norm_sq = cand_norm_sq;
                    }
                }
                if (best_level != old_level) {
                    float new_coeff = (2.0f * best_level - K) / K;
                    code_vec_ip = ip_without_dim + new_coeff * rotated_buf[i];
                    code_norm_sq = norm_sq_without_dim + new_coeff * new_coeff;
                    levels[i] = best_level;
                    changed = true;
                }
            }
            if (!changed) break;
            float cosine_sq = code_vec_ip * code_vec_ip / code_norm_sq;
            if (iter > 0 && (cosine_sq - prev_cosine_sq) < 1.0f / (K * K)) break;
            prev_cosine_sq = cosine_sq;
        }

        float code_ip = 0.0f;
        float code_parent_ip = 0.0f;
        for (size_t i = 0; i < D; ++i) {
            out_code.set_value(i, static_cast<uint8_t>(levels[i]));
            float coeff = (2.0f * levels[i] - K) / K;
            code_ip += coeff * rotated_buf[i];
            if (rotated_parent) code_parent_ip += coeff * rotated_parent[i];
        }
        if (out_code_parent_ip) *out_code_parent_ip = code_parent_ip * inv_sqrt_d_;
        return code_ip * inv_sqrt_d_;
    }
};

}
