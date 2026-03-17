#pragma once

#include <cmath>
#include <limits>
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <vector>

namespace cphnsw {

namespace evt_detail {
    static constexpr int kEmpiricalCheckpoints = 8;
    static constexpr float kCheckpointAlphas[kEmpiricalCheckpoints] =
        {0.5f, 0.1f, 0.05f, 0.01f, 0.005f, 0.001f, 5e-4f, 1e-4f};
}

struct EVTState {
    float u = 0.0f;
    float p_u = 0.0f;
    float xi = 0.0f;
    float beta = 0.0f;
    uint32_t n_tail = 0;
    bool fitted = false;
    bool use_empirical = false;
    float empirical_checkpoints[evt_detail::kEmpiricalCheckpoints] = {};
};

namespace evt_crc {


inline float evt_quantile(float alpha, const EVTState& evt) {
    if (alpha >= evt.p_u) {
        return evt.u;
    }

    if (evt.use_empirical) {
        constexpr int N = evt_detail::kEmpiricalCheckpoints;
        const float* A = evt_detail::kCheckpointAlphas;
        const float* Q = evt.empirical_checkpoints;

        for (int j = 0; j < N - 1; ++j) {
            if (alpha >= A[j + 1]) {
                float t = (alpha - A[j + 1]) / (A[j] - A[j + 1]);
                return Q[j + 1] * (1.0f - t) + Q[j] * t;
            }
        }
        if (N >= 2 && A[N-2] > A[N-1]) {
            float log_ratio = std::log(A[N-2] / A[N-1]);
            float slope = (log_ratio > std::numeric_limits<float>::epsilon())
                ? (Q[N-1] - Q[N-2]) / log_ratio : 0.0f;
            return Q[N-1] + slope * std::log(A[N-1] / alpha);
        }
        return Q[N - 1];
    }

    float ratio = evt.p_u / alpha;

    if (std::abs(evt.xi) < std::numeric_limits<float>::epsilon()) {
        return evt.u + evt.beta * std::log(ratio);
    } else {
        return evt.u + (evt.beta / evt.xi) * (std::pow(ratio, evt.xi) - 1.0f);
    }
}


inline EVTState fit_gpd(const float* sorted_abs_resid, size_t n,
                        float threshold_quantile, size_t min_tail) {
    EVTState state;

    if (n < min_tail * 2) {
        return state;
    }

    size_t u_idx = static_cast<size_t>(static_cast<float>(n) * threshold_quantile);
    u_idx = std::min(u_idx, n - 1);
    state.u = sorted_abs_resid[u_idx];

    std::vector<double> y;
    y.reserve(n - u_idx);
    for (size_t i = u_idx + 1; i < n; ++i) {
        double yi = sorted_abs_resid[i] - state.u;
        if (yi > 0.0) y.push_back(yi);
    }

    uint32_t m = static_cast<uint32_t>(y.size());
    state.n_tail = m;
    state.p_u = static_cast<float>(m) / static_cast<float>(n);

    if (m < min_tail) {
        return state;
    }

    double sum_y = 0.0, sum_y2 = 0.0;
    double y_max = 0.0;
    for (uint32_t i = 0; i < m; ++i) {
        sum_y += y[i];
        sum_y2 += y[i] * y[i];
        if (y[i] > y_max) y_max = y[i];
    }
    double mean_y = sum_y / m;

    double var_y = sum_y2 / m - mean_y * mean_y;
    double xi_mom, beta_mom;
    if (var_y < std::numeric_limits<float>::min()) {
        xi_mom = 0.0;
        beta_mom = std::max(mean_y, static_cast<double>(std::numeric_limits<float>::epsilon()));
    } else {
        xi_mom = 0.5 * (1.0 - mean_y * mean_y / var_y);
        beta_mom = mean_y * (1.0 - xi_mom);
    }

    double xi = xi_mom;
    double beta = std::max(beta_mom, static_cast<double>(std::numeric_limits<float>::epsilon()));
    bool mle_converged = false;

    for (;;) {
        if (std::abs(xi) < std::numeric_limits<float>::epsilon()) {
            beta = mean_y;
            xi = 0.0;
            mle_converged = true;
            break;
        }

        bool feasible = true;
        for (uint32_t i = 0; i < m; ++i) {
            if (1.0 + xi * y[i] / beta <= 0.0) { feasible = false; break; }
        }
        if (!feasible) break;


        double beta_new = beta;
        for (;;) {
            float beta_prev = beta_new;
            double sum_yz = 0.0;
            bool inner_feasible = true;
            for (uint32_t i = 0; i < m; ++i) {
                double z = 1.0 + xi * y[i] / beta_new;
                if (z <= 0.0) { inner_feasible = false; break; }
                sum_yz += y[i] / z;
            }
            if (!inner_feasible) break;
            beta_new = (1.0 + xi) * sum_yz / m;
            if (beta_new < std::numeric_limits<float>::epsilon()) beta_new = std::numeric_limits<float>::epsilon();
            if (std::abs(beta_new - beta_prev) < std::numeric_limits<float>::epsilon() * std::max(beta_new, static_cast<double>(std::numeric_limits<float>::epsilon()))) break;
        }
        beta = beta_new;


        double score = 0.0, info = 0.0;
        for (uint32_t i = 0; i < m; ++i) {
            double z = 1.0 + xi * y[i] / beta;
            if (z <= 0.0) { score = 0.0; break; }
            double lz = std::log(z);
            double w = y[i] / (beta * z);
            score += -lz / (xi * xi) + (1.0 + 1.0 / xi) * w;
            info += 2.0 * lz / (xi * xi * xi) - 2.0 * w / (xi * xi)
                    - (1.0 + 1.0 / xi) * w * w;
        }

        if (std::abs(info) < std::numeric_limits<float>::min()) break;
        double xi_new = xi - score / info;

        if (std::abs(xi_new - xi) < std::numeric_limits<float>::epsilon()) {
            xi = xi_new;
            mle_converged = true;
            break;
        }
        xi = xi_new;
    }

    if (!mle_converged) {
        xi = xi_mom;
        beta = beta_mom;
    }

    state.xi = static_cast<float>(xi);
    state.beta = std::max(static_cast<float>(beta), std::numeric_limits<float>::epsilon());
    state.fitted = true;
    return state;
}


inline float ks_test_gpd(const float* sorted_tail, size_t m,
                         float xi, float beta) {
    float max_diff = 0.0f;
    for (size_t i = 0; i < m; ++i) {
        float y = sorted_tail[i];
        float F_emp = static_cast<float>(i + 1) / static_cast<float>(m);
        float F_gpd;
        if (std::abs(xi) < std::numeric_limits<float>::epsilon()) {
            F_gpd = 1.0f - std::exp(-y / beta);
        } else {
            float z = 1.0f + xi * y / beta;
            F_gpd = (z > 0.0f) ? 1.0f - std::pow(z, -1.0f / xi) : 1.0f;
        }
        float diff = std::abs(F_emp - F_gpd);
        if (diff > max_diff) max_diff = diff;
    }
    return max_diff;
}

inline float ks_critical(size_t n) {
    // Lilliefors correction for 2 estimated GPD parameters (ξ, β):
    // inflates DKW critical value by (1 + p/sqrt(n)) where p=2
    float sqrt_n = std::sqrt(static_cast<float>(std::max(n, size_t(1))));
    // 1.358 = sqrt(ln(2/0.05) / 2), DKW critical value at α=0.05
    return (1.0f + 2.0f / sqrt_n) * 1.358f / sqrt_n;
}


inline EVTState fit_gpd_stable(const float* sorted_abs_resid, size_t n,
                                size_t min_tail) {
    float thresh = 1.0f - 1.0f / std::sqrt(static_cast<float>(std::max(n, size_t(4))));

    EVTState best = fit_gpd(sorted_abs_resid, n, thresh, min_tail);

    if (best.fitted && best.n_tail >= min_tail) {
        size_t u_idx = static_cast<size_t>(
            static_cast<float>(n) * thresh);
        u_idx = std::min(u_idx, n - 1);
        std::vector<float> sorted_tail;
        sorted_tail.reserve(n - u_idx);
        for (size_t i = u_idx + 1; i < n; ++i) {
            float yi = sorted_abs_resid[i] - best.u;
            if (yi > 0.0f) sorted_tail.push_back(yi);
        }
        std::sort(sorted_tail.begin(), sorted_tail.end());

        if (!sorted_tail.empty()) {
            float ks_stat = ks_test_gpd(sorted_tail.data(), sorted_tail.size(),
                                         best.xi, best.beta);
            if (ks_stat > ks_critical(sorted_tail.size())) {
                best.use_empirical = true;
                for (int j = 0; j < evt_detail::kEmpiricalCheckpoints; ++j) {
                    float target_quantile = 1.0f -
                        evt_detail::kCheckpointAlphas[j] / best.p_u;
                    target_quantile = std::clamp(target_quantile, 0.0f, 1.0f);
                    size_t idx = static_cast<size_t>(
                        target_quantile * static_cast<float>(sorted_tail.size()));
                    if (idx >= sorted_tail.size()) idx = sorted_tail.size() - 1;
                    best.empirical_checkpoints[j] = best.u + sorted_tail[idx];
                }
            }
        }
    }

    return best;
}

}
}
