#include "tune/metrics.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "tune/scale.hpp"
#include "tune/yin_offline.hpp"

namespace tune {
namespace {

// Per-sample flag: inside a scored region (target >= 0, past the settle time
// after onset, not within edge_s of the segment end).
std::vector<std::uint8_t> scored_mask(const SynthSignal& sig, double settle_s, double edge_s) {
    const std::size_t n = sig.target.size();
    std::vector<std::uint8_t> m(n, 0);
    const std::size_t settle = static_cast<std::size_t>(settle_s * sig.sample_rate);
    const std::size_t edge = static_cast<std::size_t>(edge_s * sig.sample_rate);
    std::size_t i = 0;
    while (i < n) {
        if (sig.target[i] < 0) {
            ++i;
            continue;
        }
        std::size_t j = i;
        while (j < n && sig.target[j] == sig.target[i] && !(j > i && sig.onset[j])) ++j;
        for (std::size_t k = i + settle; k + edge < j; ++k) m[k] = 1;
        i = j;
    }
    return m;
}

double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const double pos = p * (v.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(std::floor(pos));
    const std::size_t hi = std::min(lo + 1, v.size() - 1);
    return v[lo] + (v[hi] - v[lo]) * (pos - lo);
}

}  // namespace

PitchStats score_pitch(const std::vector<float>& audio, const SynthSignal& sig, int delay,
                       double settle_s, double edge_s) {
    const auto mask = scored_mask(sig, settle_s, edge_s);
    const auto frames = yin_track(audio, sig.sample_rate, 60.0, 1000.0);
    std::vector<double> errs;
    PitchStats s;
    int gross = 0, octave = 0, within = 0;
    for (const auto& f : frames) {
        const long src = std::lround(f.t * sig.sample_rate) - delay;  // input-time sample
        if (src < 0 || src >= static_cast<long>(mask.size()) || !mask[src]) continue;
        const int target = sig.target[src];
        double err;
        if (f.hz <= 0.0) {
            err = 1200.0;  // unvoiced where voice is expected counts as gross
        } else {
            err = 1200.0 * std::log2(f.hz / midi_to_hz(target));
        }
        const double a = std::abs(err);
        errs.push_back(a);
        if (a > 50.0) ++gross;
        if (a > 1000.0) ++octave;
        if (a <= 10.0) ++within;
    }
    s.frames = static_cast<int>(errs.size());
    if (s.frames > 0) {
        s.median_abs_cents = percentile(errs, 0.5);
        s.p95_abs_cents = percentile(errs, 0.95);
        s.within10_pct = 100.0 * within / s.frames;
        s.gross50_pct = 100.0 * gross / s.frames;
        s.octave_pct = 100.0 * octave / s.frames;
    }
    return s;
}

double hf_energy_db(const std::vector<float>& audio, const SynthSignal& sig, int delay,
                    double cut_hz) {
    const auto mask = scored_mask(sig, 0.0, 0.0);
    const double fs = sig.sample_rate;
    // 4th-order Butterworth-ish high-pass via two biquads.
    struct BQ {
        double b0, b1, b2, a1, a2, z1 = 0, z2 = 0;
        double p(double x) {
            const double y = b0 * x + z1;
            z1 = b1 * x - a1 * y + z2;
            z2 = b2 * x - a2 * y;
            return y;
        }
    };
    auto hp = [&](double q) {
        const double w = 2 * 3.14159265358979323846 * cut_hz / fs, cw = std::cos(w);
        const double alpha = std::sin(w) / (2 * q), a0 = 1 + alpha;
        return BQ{(1 + cw) / 2 / a0, -(1 + cw) / a0, (1 + cw) / 2 / a0, -2 * cw / a0,
                  (1 - alpha) / a0};
    };
    BQ s1 = hp(0.5412), s2 = hp(1.3066);
    double e_hf = 0.0, e_all = 0.0;
    for (std::size_t i = 0; i < audio.size(); ++i) {
        const double y = s2.p(s1.p(audio[i]));
        const long src = static_cast<long>(i) - delay;
        if (src < 0 || src >= static_cast<long>(mask.size()) || !mask[src]) continue;
        e_hf += y * y;
        e_all += static_cast<double>(audio[i]) * audio[i];
    }
    if (e_all <= 0.0) return -200.0;
    return 10.0 * std::log10(e_hf / e_all + 1e-30);
}

std::string format_stats(const PitchStats& s) {
    char buf[256];
    std::snprintf(buf, sizeof buf, "%6d | %7.1f | %7.1f | %6.1f%% | %6.2f%% | %5.2f%%", s.frames,
                  s.median_abs_cents, s.p95_abs_cents, s.within10_pct, s.gross50_pct,
                  s.octave_pct);
    return buf;
}

}  // namespace tune
