#pragma once

#include "core.hpp"
#include "segmentation.hpp"
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
// Fast Hadamard Transform
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
// Random Hadamard Rotation
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
        // Initialize with uniform bit allocation (pre-segmentation)
        for (size_t i = 0; i < D; ++i) {
            k_int_[i] = K_INT;
            k_float_[i] = K;
        }
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
        compute_segmentation_from_data(vecs, num_vecs);

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

    const std::vector<Segment>& get_segments() const { return segments_; }
    const std::array<float, D>& get_dim_variance() const { return dim_variance_; }

    QueryType encode_query_raw(const float* vec) const {
        thread_local AlignedVector<float> buf(padded_dim_);
        thread_local AlignedVector<uint8_t> quantized_query(padded_dim_);
        buf.resize(padded_dim_);
        quantized_query.resize(padded_dim_);

        QueryType query;
        rotate_and_normalize(vec, buf.data());
        build_lut(buf.data(), quantized_query.data(), query);
        return query;
    }

    void rotate_and_normalize(const float* input, float* output) const {
        rotation_.apply_copy(input, output);
#ifdef __AVX512F__
        __m512 vnf = _mm512_set1_ps(norm_factor_);
        for (size_t i = 0; i < padded_dim_; i += 16)
            _mm512_storeu_ps(output + i, _mm512_mul_ps(_mm512_loadu_ps(output + i), vnf));
#else
        __m256 vnf = _mm256_set1_ps(norm_factor_);
        for (size_t i = 0; i < padded_dim_; i += 8)
            _mm256_storeu_ps(output + i, _mm256_mul_ps(_mm256_loadu_ps(output + i), vnf));
#endif
    }

    void rotate_raw_vector(const float* vec, float* out) const {
        rotate_and_normalize(vec, out);
    }

    void build_lut(const float* buf, uint8_t* quantized_query, QueryType& query) const {
        constexpr float LUT_LEVELS = static_cast<float>((1 << (CHAR_BIT / 2)) - 1);
        float lut_min = buf[0], lut_max = buf[0];
        for (size_t i = 1; i < padded_dim_; ++i) {
            if (buf[i] < lut_min) lut_min = buf[i];
            if (buf[i] > lut_max) lut_max = buf[i];
        }

        float delta = (lut_max - lut_min) / LUT_LEVELS;
        float inv_delta = 1.0f / delta;

        float quantized_sum = 0.0f;
        for (size_t i = 0; i < padded_dim_; ++i) {
            float val = (buf[i] - lut_min) * inv_delta;
            int u = static_cast<int>(val + 0.5f);
            if (u < 0) u = 0;
            if (u > static_cast<int>(LUT_LEVELS)) u = static_cast<int>(LUT_LEVELS);
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

    const std::vector<float>& get_centroid() const { return centroid_; }
    void set_centroid(std::vector<float> c) { centroid_ = std::move(c); }

    std::pair<NbitCodeStorage<D, BitWidth>, VertexAuxData> compute_neighbor_aux_nbit(
        const float* parent_vec, const float* neighbor_vec,
        const float* rotated_parent) const
    {
        NbitCodeStorage<D, BitWidth> result_code;
        VertexAuxData result_aux;
        result_code.clear();

        alignas(SIMD_ALIGNMENT) float diff[D];
        float centered_norm_sq = 0.0f;
        for (size_t i = 0; i < dim_; ++i) {
            diff[i] = neighbor_vec[i] - parent_vec[i];
            centered_norm_sq += diff[i] * diff[i];
        }
        for (size_t i = dim_; i < D; ++i) diff[i] = 0.0f;

        float centered_norm = std::sqrt(centered_norm_sq);
        result_aux.centered_norm = centered_norm;

        float inv_centered_norm = 1.0f / centered_norm;
        for (size_t i = 0; i < D; ++i) diff[i] *= inv_centered_norm;

        alignas(SIMD_ALIGNMENT) float rotated[D];
        rotate_and_normalize(diff, rotated);

        float code_parent_ip = 0.0f;
        result_aux.code_ip = caq_quantize(rotated, result_code,
                                         rotated_parent, &code_parent_ip);
        result_aux.code_parent_ip = code_parent_ip;
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
        code.centered_norm = norm;

        float inv_norm = 1.0f / norm;
        for (size_t i = 0; i < dim_; ++i) centered_buf[i] *= inv_norm;

        rotate_and_normalize(centered_buf, buf);

        code.code_ip = caq_quantize(buf, code.codes);
        return code;
    }

private:
    size_t dim_;
    RandomHadamardRotation rotation_;
    size_t padded_dim_;
    float norm_factor_;
    float inv_sqrt_d_;
    std::vector<float> centroid_;
    std::vector<Segment> segments_;
    std::array<float, D> dim_variance_{};
    std::array<int, D> k_int_;
    std::array<float, D> k_float_;

    void compute_segmentation_from_data(const float* vecs, size_t num_vecs) {
        alignas(SIMD_ALIGNMENT) float var[D] = {};
        AlignedVector<float> buf(padded_dim_);
        std::vector<float> centered(dim_);

        for (size_t i = 0; i < num_vecs; ++i) {
            float norm_sq = 0.0f;
            for (size_t j = 0; j < dim_; ++j) {
                float v = vecs[i * dim_ + j] - centroid_[j];
                centered[j] = v;
                norm_sq += v * v;
            }
            float inv_norm = 1.0f / std::sqrt(norm_sq);
            for (size_t j = 0; j < dim_; ++j) centered[j] *= inv_norm;

            rotate_and_normalize(centered.data(), buf.data());

            for (size_t j = 0; j < D; ++j) var[j] += buf[j] * buf[j];
        }

        float inv_n = 1.0f / static_cast<float>(num_vecs);
        for (size_t j = 0; j < D; ++j) {
            var[j] *= inv_n;
            dim_variance_[j] = var[j];
        }

        segments_ = compute_segmentation<D>(var, BitWidth * D);

        for (const auto& seg : segments_) {
            for (size_t i = seg.start; i < seg.start + seg.len; ++i) {
                k_int_[i] = (1 << seg.bits) - 1;
                k_float_[i] = static_cast<float>(k_int_[i]);
            }
        }
    }

    float caq_quantize(const float* rotated_buf,
                       NbitCodeStorage<D, BitWidth>& out_code,
                       const float* rotated_parent = nullptr,
                       float* out_code_parent_ip = nullptr) const {
        const size_t pd = padded_dim_;

        float buf_min = rotated_buf[0], buf_max = rotated_buf[0];
        for (size_t i = 1; i < pd; ++i) {
            if (rotated_buf[i] < buf_min) buf_min = rotated_buf[i];
            if (rotated_buf[i] > buf_max) buf_max = rotated_buf[i];
        }
        float range = buf_max - buf_min;
        float inv_range = 1.0f / range;

        thread_local std::vector<int> levels;
        levels.resize(pd);

        float code_vec_ip = 0.0f, code_norm_sq = 0.0f;
        for (size_t i = 0; i < pd; ++i) {
            float kf = k_float_[i];
            float val = (rotated_buf[i] - buf_min) * inv_range * kf;
            int u = static_cast<int>(val + 0.5f);
            if (u < 0) u = 0;
            if (u > k_int_[i]) u = k_int_[i];
            levels[i] = u;
            float coeff = (2.0f * u - kf) / kf;
            code_vec_ip += c * rotated_buf[i];
            code_norm_sq += c * c;
        }

        float prev_cosine_sq = 0.0f;
        for (size_t iter = 0; ; ++iter) {
            bool changed = false;
            for (size_t i = 0; i < pd; ++i) {
                float kf = k_float_[i];
                int ki = k_int_[i];
                int old_level = levels[i];
                float old_coeff = (2.0f * old_level - kf) / kf;
                float ip_without_dim = code_vec_ip - old_coeff * rotated_buf[i];
                float norm_sq_without_dim = code_norm_sq - old_coeff * old_coeff;
                int best_level = old_level;
                float best_ip = code_vec_ip, best_norm_sq = code_norm_sq;
                // K for the largest allowed bit width; delta search above, exhaustive below
                constexpr int DELTA_SEARCH_K = (1 << ALLOWED_BITS[NUM_ALLOWED_BITS - 1]) - 1;
                if (ki >= DELTA_SEARCH_K) {
                    for (int d : {-1, +1}) {
                        int try_level = old_level + d;
                        if (try_level < 0 || try_level > ki) continue;
                        float coeff = (2.0f * try_level - kf) / kf;
                        float cand_ip = ip_without_dim + coeff * rotated_buf[i];
                        float cand_norm_sq = norm_sq_without_dim + coeff * coeff;
                        if (cand_ip * cand_ip * best_norm_sq > best_ip * best_ip * cand_norm_sq) {
                            best_level = try_level;
                            best_ip = cand_ip;
                            best_norm_sq = cand_norm_sq;
                        }
                    }
                } else {
                    // 1-2 bit: exhaustive search
                    for (int u = 0; u <= ki; ++u) {
                        if (u == old_level) continue;
                        float coeff = (2.0f * u - kf) / kf;
                        float cand_ip = ip_without_dim + coeff * rotated_buf[i];
                        float cand_norm_sq = norm_sq_without_dim + coeff * coeff;
                        if (cand_ip * cand_ip * best_norm_sq > best_ip * best_ip * cand_norm_sq) {
                            best_level = u;
                            best_ip = cand_ip;
                            best_norm_sq = cand_norm_sq;
                        }
                    }
                }
                if (best_level != old_level) {
                    float new_coeff = (2.0f * best_level - kf) / kf;
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
        for (size_t i = 0; i < pd; ++i) {
            float kf = k_float_[i];
            out_code.set_value(i, static_cast<uint8_t>(levels[i]));
            float coeff = (2.0f * levels[i] - kf) / kf;
            code_ip += c * rotated_buf[i];
            if (rotated_parent) code_parent_ip += coeff * rotated_parent[i];
        }
        if (out_code_parent_ip) *out_code_parent_ip = code_parent_ip * inv_sqrt_d_;
        return code_ip * inv_sqrt_d_;
    }
};

}
