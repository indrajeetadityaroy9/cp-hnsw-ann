#pragma once

#include <cstddef>
#include <cstdint>
#include <climits>
#include <vector>
#include <limits>

namespace evtq {

struct Segment {
    size_t start;
    size_t len;
    size_t bits;
};

static constexpr size_t SEGMENT_ALIGN = sizeof(uint64_t) * CHAR_BIT;
static constexpr size_t ALLOWED_BITS[] = {1, 2, 4};
static constexpr size_t NUM_ALLOWED_BITS = 3;

template <size_t D>
std::vector<Segment> compute_segmentation(
    const float* per_dim_variance, size_t total_bit_budget)
{
    if constexpr (D < SEGMENT_ALIGN) {
        size_t bits = total_bit_budget / D;
        return {{0, D, bits}};
    }

    std::vector<float> prefix_var(D + 1, 0.0f);
    for (size_t i = 0; i < D; ++i) {
        prefix_var[i + 1] = prefix_var[i] + per_dim_variance[i];
    }

    auto range_var = [&](size_t start, size_t end) -> float {
        return prefix_var[end] - prefix_var[start];
    };

    size_t num_d_steps = D / SEGMENT_ALIGN + 1;
    size_t num_b_steps = total_bit_budget + 1;

    std::vector<float> dp(num_d_steps * num_b_steps, std::numeric_limits<float>::max());
    struct Choice { size_t len; size_t bits; };
    std::vector<Choice> choice(num_d_steps * num_b_steps, {0, 0});

    auto idx = [&](size_t d_step, size_t b) -> size_t {
        return d_step * num_b_steps + b;
    };

    dp[idx(0, 0)] = 0.0f;

    for (size_t d_step = 0; d_step < num_d_steps - 1; ++d_step) {
        size_t d = d_step * SEGMENT_ALIGN;
        for (size_t b = 0; b <= total_bit_budget; ++b) {
            if (dp[idx(d_step, b)] == std::numeric_limits<float>::max()) continue;
            float cur = dp[idx(d_step, b)];

            for (size_t seg_len = SEGMENT_ALIGN; d + seg_len <= D; seg_len += SEGMENT_ALIGN) {
                size_t next_d_step = (d + seg_len) / SEGMENT_ALIGN;
                float sv = range_var(d, d + seg_len);

                for (size_t bi = 0; bi < NUM_ALLOWED_BITS; ++bi) {
                    size_t seg_bits = ALLOWED_BITS[bi];
                    size_t bits_used = seg_len * seg_bits;
                    size_t next_b = b + bits_used;
                    if (next_b > total_bit_budget) continue;

                    float scale = static_cast<float>(1u << (2 * seg_bits));
                    float err = cur + sv / scale;
                    if (err < dp[idx(next_d_step, next_b)]) {
                        dp[idx(next_d_step, next_b)] = err;
                        choice[idx(next_d_step, next_b)] = {seg_len, seg_bits};
                    }
                }
            }
        }
    }

    size_t final_d_step = D / SEGMENT_ALIGN;
    std::vector<Segment> segments;

    size_t cur_b = total_bit_budget;
    size_t cur_d_step = final_d_step;
    while (cur_d_step > 0) {
        auto& ch = choice[idx(cur_d_step, cur_b)];
        size_t d = cur_d_step * SEGMENT_ALIGN;
        segments.push_back({d - ch.len, ch.len, ch.bits});
        cur_b -= ch.len * ch.bits;
        cur_d_step = (d - ch.len) / SEGMENT_ALIGN;
    }

    std::reverse(segments.begin(), segments.end());
    return segments;
}

}
