#pragma once

#include "core.hpp"
#include <array>
#include <cmath>
#include <random>
#include <utility>
#include <vector>
#include <omp.h>

namespace evtq {

inline void fht(float* vec, size_t len) {
#ifdef __AVX512F__
    constexpr size_t SIMD_WIDTH = 16;
    for (size_t i = 0; i < len; i += SIMD_WIDTH) {
        __m512 v = _mm512_loadu_ps(&vec[i]);
        __m512 s1 = _mm512_permute_ps(v, 0b10110001);
        v = _mm512_mask_blend_ps(0xAAAA, _mm512_add_ps(v, s1), _mm512_sub_ps(v, s1));
        __m512 s2 = _mm512_permute_ps(v, 0b01001110);
        v = _mm512_mask_blend_ps(0xCCCC, _mm512_add_ps(v, s2), _mm512_sub_ps(v, s2));
        __m512 s4 = _mm512_shuffle_f32x4(v, v, _MM_SHUFFLE(2, 3, 0, 1));
        v = _mm512_mask_blend_ps(0xF0F0, _mm512_add_ps(v, s4), _mm512_sub_ps(v, s4));
        __m512 s8 = _mm512_shuffle_f32x4(v, v, _MM_SHUFFLE(1, 0, 3, 2));
        v = _mm512_mask_blend_ps(0xFF00, _mm512_add_ps(v, s8), _mm512_sub_ps(v, s8));
        _mm512_storeu_ps(&vec[i], v);
    }
    for (size_t h = SIMD_WIDTH; h < len; h *= 2)
        for (size_t i = 0; i < len; i += h * 2)
            for (size_t j = i; j < i + h; j += SIMD_WIDTH) {
                __m512 x = _mm512_loadu_ps(&vec[j]), y = _mm512_loadu_ps(&vec[j + h]);
                _mm512_storeu_ps(&vec[j], _mm512_add_ps(x, y));
                _mm512_storeu_ps(&vec[j + h], _mm512_sub_ps(x, y));
            }
#else
    constexpr size_t SIMD_WIDTH = 8;
    for (size_t i = 0; i < len; i += SIMD_WIDTH) {
        __m256 v = _mm256_loadu_ps(&vec[i]);
        __m256 s1 = _mm256_permute_ps(v, 0b10110001);
        v = _mm256_blend_ps(_mm256_add_ps(v, s1), _mm256_sub_ps(v, s1), 0b10101010);
        __m256 s2 = _mm256_permute_ps(v, 0b01001110);
        v = _mm256_blend_ps(_mm256_add_ps(v, s2), _mm256_sub_ps(v, s2), 0b11001100);
        __m256 s4 = _mm256_permute2f128_ps(v, v, 0x01);
        v = _mm256_blend_ps(_mm256_add_ps(v, s4), _mm256_sub_ps(v, s4), 0b11110000);
        _mm256_storeu_ps(&vec[i], v);
    }
    for (size_t h = SIMD_WIDTH; h < len; h *= 2)
        for (size_t i = 0; i < len; i += h * 2)
            for (size_t j = i; j < i + h; j += SIMD_WIDTH) {
                __m256 x = _mm256_loadu_ps(&vec[j]), y = _mm256_loadu_ps(&vec[j + h]);
                _mm256_storeu_ps(&vec[j], _mm256_add_ps(x, y));
                _mm256_storeu_ps(&vec[j + h], _mm256_sub_ps(x, y));
            }
#endif
}

struct RandomHadamardRotation {
    static constexpr size_t NUM_LAYERS = 3;
    size_t original_dim_, padded_dim_;
    std::array<AlignedVector<float>, NUM_LAYERS> signs_float_;

    explicit RandomHadamardRotation(size_t dim, uint64_t seed) : original_dim_(dim), padded_dim_(std::bit_ceil(dim)) {
        std::mt19937_64 rng(seed);
        std::uniform_int_distribution<int> coin(0, 1);
        for (size_t layer = 0; layer < NUM_LAYERS; ++layer) {
            signs_float_[layer].resize(padded_dim_);
            for (size_t i = 0; i < padded_dim_; ++i) signs_float_[layer][i] = coin(rng) ? 1.0f : -1.0f;
        }
    }

    void apply(float* x) const {
        for (size_t layer = 0; layer < NUM_LAYERS; ++layer) {
            apply_diagonal(x, signs_float_[layer].data());
            fht(x, padded_dim_);
        }
    }

    void apply_copy(const float* input, float* output) const {
        std::memcpy(output, input, original_dim_ * sizeof(float));
        std::memset(output + original_dim_, 0, (padded_dim_ - original_dim_) * sizeof(float));
        apply(output);
    }

    void apply_diagonal(float* x, const float* signs_f) const {
#ifdef __AVX512F__
        for (size_t i = 0; i < padded_dim_; i += 16)
            _mm512_storeu_ps(x + i, _mm512_mul_ps(_mm512_loadu_ps(x + i), _mm512_load_ps(signs_f + i)));
#else
        for (size_t i = 0; i < padded_dim_; i += 8)
            _mm256_storeu_ps(x + i, _mm256_mul_ps(_mm256_loadu_ps(x + i), _mm256_load_ps(signs_f + i)));
#endif
    }
};

template <size_t D>
struct NbitRaBitQEncoder {
    using CodeType = NbitRaBitQCode<D>;
    using QueryType = RaBitQQuery<D>;
    static constexpr size_t NUM_SUB_SEGMENTS = num_sub_segments<D>;
    static constexpr int K_INT = (1 << BIT_WIDTH) - 1;
    static constexpr float K = static_cast<float>(K_INT);

    size_t dim_;
    RandomHadamardRotation rotation_;
    float norm_factor_, inv_sqrt_d_, dot_slack_;
    std::vector<float> centroid_;

    explicit NbitRaBitQEncoder(size_t dim, uint64_t rotation_seed) : dim_(dim), rotation_(dim, rotation_seed), centroid_(dim, 0.0f) {
        float d = static_cast<float>(D);
        norm_factor_ = 1.0f / (d * std::sqrt(d));
        inv_sqrt_d_ = 1.0f / std::sqrt(d);
        dot_slack_ = std::sqrt((static_cast<float>(M_PI) / 2.0f - 1.0f) / static_cast<float>(K_INT * K_INT * D));
    }

    void encode_batch(const float* vecs, size_t num_vecs, CodeType* codes) {
        compute_centroid(vecs, num_vecs);
        #pragma omp parallel
        {
            AlignedVector<float> buf(D), centered(D);
            #pragma omp for schedule(static)
            for (size_t i = 0; i < num_vecs; ++i) codes[i] = encode_impl(vecs + i * dim_, buf.data(), centered.data());
        }
    }

    QueryType encode_query_raw(const float* vec) const {
        alignas(SIMD_ALIGNMENT) float buf[D];
        alignas(SIMD_ALIGNMENT) uint8_t quantized_query[D];
        QueryType query;
        rotate_and_normalize(vec, buf);
        build_lut(buf, quantized_query, query);
        query.dot_slack = dot_slack_;
        return query;
    }

    void rotate_and_normalize(const float* input, float* output) const {
        rotation_.apply_copy(input, output);
#ifdef __AVX512F__
        __m512 vnf = _mm512_set1_ps(norm_factor_);
        for (size_t i = 0; i < D; i += 16) _mm512_storeu_ps(output + i, _mm512_mul_ps(_mm512_loadu_ps(output + i), vnf));
#else
        __m256 vnf = _mm256_set1_ps(norm_factor_);
        for (size_t i = 0; i < D; i += 8) _mm256_storeu_ps(output + i, _mm256_mul_ps(_mm256_loadu_ps(output + i), vnf));
#endif
    }

    const std::vector<float>& get_centroid() const { return centroid_; }
    void set_centroid(std::vector<float> c) { centroid_ = std::move(c); }

    std::pair<NbitCodeStorage<D>, VertexAuxData> compute_neighbor_aux_nbit(const float* parent_vec, const float* neighbor_vec, const float* rotated_parent) const {
        NbitCodeStorage<D> result_code;
        result_code.clear();
        alignas(SIMD_ALIGNMENT) float diff[D];
        float cn = subtract_and_normalize(neighbor_vec, parent_vec, diff);
        alignas(SIMD_ALIGNMENT) float rotated[D];
        rotate_and_normalize(diff, rotated);
        float code_parent_ip = 0.0f;
        float ci = caq_quantize(rotated, result_code, rotated_parent, &code_parent_ip);
        return {std::move(result_code), {cn, ci, code_parent_ip}};
    }

    void compute_centroid(const float* vecs, size_t num_vecs) {
        centroid_.assign(dim_, 0.0f);
        for (size_t i = 0; i < num_vecs; ++i)
            for (size_t j = 0; j < dim_; ++j) centroid_[j] += vecs[i * dim_ + j];
        float inv_n = 1.0f / static_cast<float>(num_vecs);
        for (size_t j = 0; j < dim_; ++j) centroid_[j] *= inv_n;
    }

    void build_lut(const float* buf, uint8_t* quantized_query, QueryType& query) const {
        float lut_min = buf[0], lut_max = buf[0];
        for (size_t i = 1; i < D; ++i) {
            if (buf[i] < lut_min) lut_min = buf[i];
            if (buf[i] > lut_max) lut_max = buf[i];
        }
        float delta = (lut_max - lut_min) / K, inv_delta = 1.0f / delta;

        float quantized_sum = 0.0f;
        for (size_t i = 0; i < D; ++i) {
            int u = static_cast<int>((buf[i] - lut_min) * inv_delta + 0.5f);
            quantized_query[i] = static_cast<uint8_t>(u);
            quantized_sum += static_cast<float>(u);
        }

        for (size_t j = 0; j < NUM_SUB_SEGMENTS; ++j)
            for (uint8_t p = 0; p < 16; ++p) {
                uint8_t sum = 0;
                for (size_t b = 0; b < 4; ++b) {
                    size_t idx = j * 4 + b;
                    if (idx < D && (p & (1u << b))) sum += quantized_query[idx];
                }
                query.lut[j][p] = sum;
            }

        query.coeff_fastscan = 2.0f * delta * inv_sqrt_d_;
        query.coeff_popcount = 2.0f * lut_min * inv_sqrt_d_;
        query.coeff_constant = -(static_cast<float>(D) * lut_min + delta * quantized_sum) * inv_sqrt_d_;
    }

    CodeType encode_impl(const float* vec, float* buf, float* centered_buf) const {
        CodeType code;
        code.clear();
        code.centered_norm = subtract_and_normalize(vec, centroid_.data(), centered_buf);
        rotate_and_normalize(centered_buf, buf);
        code.code_ip = caq_quantize(buf, code.codes);
        return code;
    }

    float subtract_and_normalize(const float* a, const float* b, float* out) const {
#ifdef __AVX512F__
        __m512 vsum = _mm512_setzero_ps();
        size_t i = 0;
        for (; i + 16 <= dim_; i += 16) {
            __m512 vd = _mm512_sub_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i));
            _mm512_storeu_ps(out + i, vd);
            vsum = _mm512_fmadd_ps(vd, vd, vsum);
        }
        float norm_sq = _mm512_reduce_add_ps(vsum);
        for (; i < dim_; ++i) { float d = a[i] - b[i]; out[i] = d; norm_sq += d * d; }
        for (size_t j = dim_; j < D; ++j) out[j] = 0.0f;
        float norm = std::sqrt(norm_sq), inv_norm = 1.0f / norm;
        __m512 vinv = _mm512_set1_ps(inv_norm);
        for (size_t j = 0; j < D; j += 16) _mm512_storeu_ps(out + j, _mm512_mul_ps(_mm512_loadu_ps(out + j), vinv));
#else
        __m256 vsum = _mm256_setzero_ps();
        size_t i = 0;
        for (; i + 8 <= dim_; i += 8) {
            __m256 vd = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
            _mm256_storeu_ps(out + i, vd);
            vsum = _mm256_fmadd_ps(vd, vd, vsum);
        }
        float norm_sq = hsum256(vsum);
        for (; i < dim_; ++i) { float d = a[i] - b[i]; out[i] = d; norm_sq += d * d; }
        for (size_t j = dim_; j < D; ++j) out[j] = 0.0f;
        float norm = std::sqrt(norm_sq), inv_norm = 1.0f / norm;
        __m256 vinv = _mm256_set1_ps(inv_norm);
        for (size_t j = 0; j < D; j += 8) _mm256_storeu_ps(out + j, _mm256_mul_ps(_mm256_loadu_ps(out + j), vinv));
#endif
        return norm;
    }

    float caq_quantize(const float* rotated_buf, NbitCodeStorage<D>& out_code, const float* rotated_parent = nullptr, float* out_code_parent_ip = nullptr) const {
        float buf_min = rotated_buf[0], buf_max = rotated_buf[0];
        for (size_t i = 1; i < D; ++i) {
            if (rotated_buf[i] < buf_min) buf_min = rotated_buf[i];
            if (rotated_buf[i] > buf_max) buf_max = rotated_buf[i];
        }
        float range = buf_max - buf_min, inv_range = 1.0f / range;
        int levels[D];

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
                float old_coeff = (2.0f * levels[i] - K) / K;
                float ip_without = code_vec_ip - old_coeff * rotated_buf[i];
                float norm_without = code_norm_sq - old_coeff * old_coeff;
                int best_level = levels[i];
                float best_ip = code_vec_ip, best_nsq = code_norm_sq;
                for (int d : {-1, +1}) {
                    int try_level = levels[i] + d;
                    if (try_level < 0 || try_level > K_INT) continue;
                    float c = (2.0f * try_level - K) / K;
                    float cand_ip = ip_without + c * rotated_buf[i];
                    float cand_nsq = norm_without + c * c;
                    if (cand_ip * cand_ip * best_nsq > best_ip * best_ip * cand_nsq) {
                        best_level = try_level; best_ip = cand_ip; best_nsq = cand_nsq;
                    }
                }
                if (best_level != levels[i]) {
                    float new_coeff = (2.0f * best_level - K) / K;
                    code_vec_ip = ip_without + new_coeff * rotated_buf[i];
                    code_norm_sq = norm_without + new_coeff * new_coeff;
                    levels[i] = best_level;
                    changed = true;
                }
            }
            if (!changed) break;
            float cosine_sq = code_vec_ip * code_vec_ip / code_norm_sq;
            if (iter > 0 && (cosine_sq - prev_cosine_sq) < 1.0f / (K * K)) break;
            prev_cosine_sq = cosine_sq;
        }

        float code_parent_ip = 0.0f;
        for (size_t i = 0; i < D; ++i) {
            out_code.set_value(i, static_cast<uint8_t>(levels[i]));
            if (rotated_parent) code_parent_ip += ((2.0f * levels[i] - K) / K) * rotated_parent[i];
        }
        if (out_code_parent_ip) *out_code_parent_ip = code_parent_ip * inv_sqrt_d_;
        return code_vec_ip * inv_sqrt_d_;
    }
};

}
