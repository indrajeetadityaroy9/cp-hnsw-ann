#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <immintrin.h>

namespace cphnsw {

using NodeId = uint32_t;
constexpr NodeId INVALID_NODE = 0xFFFFFFFF;
inline constexpr size_t GRAPH_DEGREE = 32;

struct SearchResult {
    NodeId id;
    float distance;

    bool operator<(const SearchResult& other) const {
        return distance < other.distance;
    }

    bool operator>(const SearchResult& other) const {
        return distance > other.distance;
    }
};

inline size_t next_power_of_two(size_t n) {
    if (n <= 1) return 1;
    return size_t(1) << (64 - __builtin_clzll(n - 1));
}

constexpr size_t CACHE_LINE_SIZE = 64;
constexpr size_t SIMD_ALIGNMENT = 64;

template <typename T, size_t Alignment = SIMD_ALIGNMENT>
class AlignedAllocator {
public:
    using value_type = T;

    template <typename U>
    struct rebind { using other = AlignedAllocator<U, Alignment>; };

    AlignedAllocator() noexcept = default;

    template <typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    T* allocate(std::size_t n) {
        size_t bytes = n * sizeof(T);
        bytes = ((bytes + Alignment - 1) / Alignment) * Alignment;
        void* ptr = std::aligned_alloc(Alignment, bytes);
        return static_cast<T*>(ptr);
    }

    void deallocate(T* p, std::size_t) noexcept {
        std::free(p);
    }

    template <typename U>
    bool operator==(const AlignedAllocator<U, Alignment>&) const noexcept { return true; }
    template <typename U>
    bool operator!=(const AlignedAllocator<U, Alignment>&) const noexcept { return false; }
};

template <typename T, size_t Alignment = SIMD_ALIGNMENT>
using AlignedVector = std::vector<T, AlignedAllocator<T, Alignment>>;

template <int Locality = 3>
inline void prefetch_t(const void* addr) {
    __builtin_prefetch(addr, 0, Locality);
}

inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

template <size_t D>
inline float l2_distance_simd(const float* __restrict__ a, const float* __restrict__ b) {
#ifdef __AVX512F__
    __m512 sum0 = _mm512_setzero_ps();
    __m512 sum1 = _mm512_setzero_ps();
    size_t i = 0;
    for (; i + 32 <= D; i += 32) {
        __m512 d0 = _mm512_sub_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i));
        __m512 d1 = _mm512_sub_ps(_mm512_loadu_ps(a + i + 16), _mm512_loadu_ps(b + i + 16));
        sum0 = _mm512_fmadd_ps(d0, d0, sum0);
        sum1 = _mm512_fmadd_ps(d1, d1, sum1);
    }
    for (; i + 16 <= D; i += 16) {
        __m512 d = _mm512_sub_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i));
        sum0 = _mm512_fmadd_ps(d, d, sum0);
    }
    float result = _mm512_reduce_add_ps(_mm512_add_ps(sum0, sum1));
    for (; i + 8 <= D; i += 8) {
        __m256 d = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        result += hsum256(_mm256_mul_ps(d, d));
    }
    return result;
#else
    // 4 accumulators to hide FMA latency (4-cycle on modern x86)
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 32 <= D; i += 32) {
        __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8));
        __m256 d2 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16));
        __m256 d3 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24));
        sum0 = _mm256_fmadd_ps(d0, d0, sum0);
        sum1 = _mm256_fmadd_ps(d1, d1, sum1);
        sum2 = _mm256_fmadd_ps(d2, d2, sum2);
        sum3 = _mm256_fmadd_ps(d3, d3, sum3);
    }
    for (; i + 16 <= D; i += 16) {
        __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8));
        sum0 = _mm256_fmadd_ps(d0, d0, sum0);
        sum1 = _mm256_fmadd_ps(d1, d1, sum1);
    }
    for (; i + 8 <= D; i += 8) {
        __m256 d = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        sum0 = _mm256_fmadd_ps(d, d, sum0);
    }
    return hsum256(_mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3)));
#endif
}

template <size_t D>
inline float dot_product_simd(const float* __restrict__ a, const float* __restrict__ b) {
#ifdef __AVX512F__
    __m512 sum0 = _mm512_setzero_ps();
    __m512 sum1 = _mm512_setzero_ps();
    size_t i = 0;
    for (; i + 32 <= D; i += 32) {
        sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), sum0);
        sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 16), _mm512_loadu_ps(b + i + 16), sum1);
    }
    for (; i + 16 <= D; i += 16) {
        sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), sum0);
    }
    float result = _mm512_reduce_add_ps(_mm512_add_ps(sum0, sum1));
    for (; i + 8 <= D; i += 8) {
        result += hsum256(_mm256_mul_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    }
    return result;
#else
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 32 <= D; i += 32) {
        sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), sum0);
        sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), sum1);
        sum2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), sum2);
        sum3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), sum3);
    }
    for (; i + 16 <= D; i += 16) {
        sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), sum0);
        sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), sum1);
    }
    for (; i + 8 <= D; i += 8) {
        sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), sum0);
    }
    return hsum256(_mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3)));
#endif
}

template <size_t D>
constexpr size_t num_sub_segments = (D + 3) / 4;

template <size_t Bits>
struct alignas(SIMD_ALIGNMENT) BinaryCodeStorage {
    static constexpr size_t NUM_BITS = Bits;
    static constexpr size_t NUM_WORDS = (Bits + 63) / 64;

    uint64_t signs[NUM_WORDS];

    void clear() {
        std::memset(signs, 0, sizeof(signs));
    }

    void set_bit(size_t idx, bool value) {
        if (value) {
            signs[idx / 64] |= (1ULL << (idx % 64));
        } else {
            signs[idx / 64] &= ~(1ULL << (idx % 64));
        }
    }

    bool get_bit(size_t idx) const {
        return (signs[idx / 64] >> (idx % 64)) & 1;
    }

    uint32_t popcount() const {
        uint32_t count = 0;
        for (size_t i = 0; i < NUM_WORDS; ++i) {
            count += static_cast<uint32_t>(__builtin_popcountll(signs[i]));
        }
        return count;
    }
};

struct VertexAuxData {
    float centered_norm;
    float code_ip;
    float code_parent_ip;
};

template <size_t D>
struct RaBitQQuery {
    static constexpr size_t DIMS = D;
    static constexpr size_t NUM_SUB_SEGMENTS = num_sub_segments<D>;

    alignas(SIMD_ALIGNMENT) uint8_t lut[NUM_SUB_SEGMENTS][16];

    float coeff_fastscan;
    float coeff_popcount;
    float coeff_constant;

    float dot_slack = 0.0f;
};


template <size_t D, size_t BitWidth>
struct alignas(SIMD_ALIGNMENT) NbitCodeStorage {
    static constexpr size_t NUM_BITS = D;
    static constexpr size_t BIT_WIDTH = BitWidth;
    static constexpr size_t NUM_WORDS = (D + 63) / 64;

    uint64_t planes[BitWidth][NUM_WORDS];

    void clear() { std::memset(planes, 0, sizeof(planes)); }

    void set_value(size_t idx, uint8_t value) {
        size_t word = idx / 64;
        size_t bit = idx % 64;
        for (size_t b = 0; b < BitWidth; ++b) {
            if ((value >> (BitWidth - 1 - b)) & 1)
                planes[b][word] |= (1ULL << bit);
            else
                planes[b][word] &= ~(1ULL << bit);
        }
    }

    uint32_t msb_popcount() const {
        uint32_t count = 0;
        for (size_t i = 0; i < NUM_WORDS; ++i)
            count += static_cast<uint32_t>(__builtin_popcountll(planes[0][i]));
        return count;
    }

    uint32_t weighted_popcount() const {
        uint32_t total = 0;
        for (size_t b = 0; b < BitWidth; ++b) {
            uint32_t pc = 0;
            for (size_t i = 0; i < NUM_WORDS; ++i)
                pc += static_cast<uint32_t>(__builtin_popcountll(planes[b][i]));
            total += pc * (1u << (BitWidth - 1 - b));
        }
        return total;
    }
};

template <size_t D, size_t BitWidth = 1>
struct NbitRaBitQCode {
    static constexpr size_t DIMS = D;
    static constexpr size_t BIT_WIDTH = BitWidth;

    NbitCodeStorage<D, BitWidth> codes;
    float centered_norm;
    float code_ip;

    void clear() {
        codes.clear();
        centered_norm = 0.0f;
        code_ip = 0.0f;
    }
};

}
