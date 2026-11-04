#include "tune/yin_offline.hpp"

#include <algorithm>
#include <cmath>

namespace tune {

std::vector<PitchFrame> yin_track(const std::vector<float>& x, double fs, double min_hz,
                                  double max_hz, double hop_s, double threshold) {
    const int tau_max = static_cast<int>(std::ceil(fs / min_hz));
    const int tau_min = static_cast<int>(std::floor(fs / max_hz));
    const int win = tau_max + 64;  // integration window
    const int hop = static_cast<int>(std::lround(hop_s * fs));
    std::vector<PitchFrame> out;
    std::vector<double> d(static_cast<std::size_t>(tau_max) + 2);
    const long n = static_cast<long>(x.size());
    for (long start = 0; start + win + tau_max + 1 < n; start += hop) {
        // Difference function.
        for (int tau = 1; tau <= tau_max + 1; ++tau) {
            double acc = 0.0;
            for (int j = 0; j < win; ++j) {
                const double diff = static_cast<double>(x[start + j]) - x[start + j + tau];
                acc += diff * diff;
            }
            d[tau] = acc;
        }
        // Cumulative mean normalized difference.
        d[0] = 1.0;
        double running = 0.0;
        for (int tau = 1; tau <= tau_max + 1; ++tau) {
            running += d[tau];
            d[tau] = running > 0.0 ? d[tau] * tau / running : 1.0;
        }
        // Absolute threshold, then walk to the local minimum.
        int best = -1;
        for (int tau = std::max(tau_min, 2); tau <= tau_max; ++tau) {
            if (d[tau] < threshold) {
                while (tau + 1 <= tau_max && d[tau + 1] < d[tau]) ++tau;
                best = tau;
                break;
            }
        }
        PitchFrame f;
        f.t = (static_cast<double>(start) + win / 2.0) / fs;
        if (best > 0) {
            // Parabolic interpolation around the minimum.
            const double a = d[best - 1], b = d[best], c = d[best + 1];
            const double denom = a - 2 * b + c;
            const double shift = std::abs(denom) > 1e-12 ? 0.5 * (a - c) / denom : 0.0;
            f.hz = fs / (best + shift);
            f.aperiodicity = b;
        }
        out.push_back(f);
    }
    return out;
}

}  // namespace tune
