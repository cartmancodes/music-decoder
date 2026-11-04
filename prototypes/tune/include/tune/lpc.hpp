// Linear prediction helpers (spec §6.5): autocorrelation with lag window and
// white-noise correction, and Levinson-Durbin giving reflection (PARCOR)
// coefficients. Reflection coefficients are what the formant lattices use:
// any interpolation between two valid sets stays valid (|k| < 1), so the
// time-varying filters stay stable.
//
// Sign convention: A(z) = 1 + sum_j a_j z^-j, and k_i = a_i^{(i)} (the last
// coefficient of the order-i predictor).
#pragma once

#include <algorithm>
#include <cmath>

namespace tune {

// Autocorrelation r[0..order] of x[0..n).
inline void autocorrelate(const double* x, int n, int order, double* r) {
    for (int lag = 0; lag <= order; ++lag) {
        double acc = 0.0;
        for (int i = lag; i < n; ++i) acc += x[i] * x[i - lag];
        r[lag] = acc;
    }
}

// Gaussian lag window (bandwidth in Hz) plus white-noise correction. The lag
// window smooths the modelled envelope so the predictor follows formants
// rather than individual harmonics of high voices.
inline void condition_autocorrelation(double* r, int order, double fs, double bandwidth_hz,
                                      double white_noise = 1e-4) {
    const double pi = 3.14159265358979323846;
    r[0] *= 1.0 + white_noise;
    for (int i = 1; i <= order; ++i) {
        const double x = 2.0 * pi * bandwidth_hz * i / fs;
        r[i] *= std::exp(-0.5 * x * x);
    }
}

// Levinson-Durbin. Writes reflection coefficients k[0..order) (k[i] is the
// order-(i+1) coefficient) and returns the final prediction error power.
// Uses `a` (size order + 1) as scratch. Returns 0 and zero coefficients for
// silent input.
inline double levinson(const double* r, int order, double* k, double* a, double* tmp,
                       double k_limit = 0.995) {
    for (int i = 0; i < order; ++i) k[i] = 0.0;
    if (!(r[0] > 1e-12)) return 0.0;
    double err = r[0];
    a[0] = 1.0;
    for (int i = 1; i <= order; ++i) a[i] = 0.0;
    for (int i = 1; i <= order; ++i) {
        double acc = r[i];
        for (int j = 1; j < i; ++j) acc += a[j] * r[i - j];
        double ki = -acc / err;
        ki = std::clamp(ki, -k_limit, k_limit);
        for (int j = 1; j < i; ++j) tmp[j] = a[j] + ki * a[i - j];
        for (int j = 1; j < i; ++j) a[j] = tmp[j];
        a[i] = ki;
        k[i - 1] = ki;
        err *= (1.0 - ki * ki);
        if (err <= 0.0) break;
    }
    return err;
}

}  // namespace tune
