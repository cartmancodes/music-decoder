// Template implementation for fit_envelope (included from metrics.hpp).
#pragma once

#include <cmath>
#include <vector>

namespace tune {
namespace detail {
inline double peak_in(const std::vector<double>& freqs, const std::vector<double>& db, double lo,
                      double hi) {
    int best = -1;
    for (int i = 0; i < static_cast<int>(freqs.size()); ++i)
        if (freqs[i] >= lo && freqs[i] <= hi && (best < 0 || db[i] > db[best])) best = i;
    if (best <= 0 || best + 1 >= static_cast<int>(freqs.size())) return best >= 0 ? freqs[best] : 0.0;
    // Parabola through three (freq, dB) points (uneven spacing allowed).
    const double x0 = freqs[best - 1], x1 = freqs[best], x2 = freqs[best + 1];
    const double y0 = db[best - 1], y1 = db[best], y2 = db[best + 1];
    const double d = (x0 - x1) * (x0 - x2) * (x1 - x2);
    if (std::abs(d) < 1e-12) return x1;
    const double A = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / d;
    const double B = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / d;
    if (A >= 0.0) return x1;
    const double xv = -B / (2 * A);
    return (xv >= x0 && xv <= x2) ? xv : x1;
}
}  // namespace detail

template <class F>
EnvelopeFit fit_envelope(const std::vector<float>& audio, double fs, std::size_t start,
                         std::size_t len, double f0, F reference, double max_hz,
                         double formant_scale) {
    const double pi = 3.14159265358979323846;
    EnvelopeFit out;
    if (start + len > audio.size() || f0 <= 0.0) return out;
    std::vector<double> freqs, meas_db, ref_db;
    for (int k = 1; k * f0 <= max_hz; ++k) {
        const double f = k * f0;
        if (f < 150.0) continue;
        double re = 0.0, im = 0.0;
        for (std::size_t i = 0; i < len; ++i) {
            const double w = 0.5 - 0.5 * std::cos(2 * pi * (i + 0.5) / len);
            const double ph = 2 * pi * f * static_cast<double>(start + i) / fs;
            re += w * audio[start + i] * std::cos(ph);
            im += w * audio[start + i] * std::sin(ph);
        }
        const double amp = std::sqrt(re * re + im * im) * k;  // undo 1/k tilt
        freqs.push_back(f);
        meas_db.push_back(20.0 * std::log10(amp + 1e-12));
        ref_db.push_back(20.0 * std::log10(reference(f) + 1e-12));
    }
    if (freqs.empty()) return out;
    double mm = 0.0, mr = 0.0;
    for (std::size_t i = 0; i < freqs.size(); ++i) {
        mm += meas_db[i];
        mr += ref_db[i];
    }
    mm /= freqs.size();
    mr /= freqs.size();
    double acc = 0.0;
    for (std::size_t i = 0; i < freqs.size(); ++i) {
        const double d = (meas_db[i] - mm) - (ref_db[i] - mr);
        acc += d * d;
    }
    out.lsd_db = std::sqrt(acc / freqs.size());
    out.f1_hz = detail::peak_in(freqs, meas_db, 450.0 * formant_scale, 950.0 * formant_scale);
    out.f2_hz = detail::peak_in(freqs, meas_db, 950.0 * formant_scale, 1600.0 * formant_scale);
    out.harmonics = static_cast<int>(freqs.size());
    return out;
}

}  // namespace tune
