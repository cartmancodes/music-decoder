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

SettleStats score_settle(const std::vector<float>& audio, const SynthSignal& sig, int delay,
                         double tol_cents, double hold_s) {
    const double fs = sig.sample_rate;
    const double min_hz = 60.0;
    const auto frames = yin_track(audio, fs, min_hz, 1000.0, 0.002);
    const double judge_bias_s = 0.5 * (std::ceil(fs / min_hz) + 64) / fs;
    SettleStats out;
    std::vector<double> settles;
    const std::size_t n = sig.target.size();
    std::size_t i = 0;
    while (i < n) {
        if (sig.target[i] < 0 || !sig.onset[i]) {
            ++i;
            continue;
        }
        std::size_t j = i + 1;
        while (j < n && sig.target[j] == sig.target[i] && !sig.onset[j]) ++j;
        const double t_on = static_cast<double>(i) / fs;
        const double t_end = static_cast<double>(j) / fs;
        const int target = sig.target[i];
        ++out.segments;
        double run_start = -1.0;
        double settled_at = -1.0;
        for (const auto& f : frames) {
            const double t_in = f.t - static_cast<double>(delay) / fs;  // input time
            if (t_in < t_on) continue;
            if (t_in > t_end) break;
            const bool ok = f.hz > 0.0 &&
                            std::abs(1200.0 * std::log2(f.hz / midi_to_hz(target))) <= tol_cents;
            if (ok) {
                if (run_start < 0.0) run_start = t_in;
                if (t_in - run_start >= hold_s) {
                    settled_at = run_start;
                    break;
                }
            } else {
                run_start = -1.0;
            }
        }
        if (settled_at < 0.0) ++out.unsettled;
        else settles.push_back(std::max(0.0, settled_at - t_on - judge_bias_s) * 1000.0);
        i = j;
    }
    if (!settles.empty()) {
        out.median_ms = percentile(settles, 0.5);
        out.p90_ms = percentile(settles, 0.9);
    }
    return out;
}

PitchStats score_against_scale(const std::vector<float>& audio, double fs, unsigned scale_mask,
                               double min_hz, double max_hz, double max_aperiodicity) {
    const Scale scale{static_cast<std::uint16_t>(scale_mask & 0x0FFF)};
    const auto frames = yin_track(audio, fs, min_hz, max_hz);
    std::vector<double> errs;
    int gross = 0, within = 0;
    for (const auto& f : frames) {
        if (f.hz <= 0.0 || f.aperiodicity > max_aperiodicity) continue;
        const double m = hz_to_midi(f.hz);
        const double a = std::abs(100.0 * (m - nearest_allowed(scale, m)));
        errs.push_back(a);
        if (a > 50.0) ++gross;
        if (a <= 10.0) ++within;
    }
    PitchStats s;
    s.frames = static_cast<int>(errs.size());
    if (s.frames > 0) {
        s.median_abs_cents = percentile(errs, 0.5);
        s.p95_abs_cents = percentile(errs, 0.95);
        s.within10_pct = 100.0 * within / s.frames;
        s.gross50_pct = 100.0 * gross / s.frames;
    }
    return s;
}

VibratoStats score_vibrato(const std::vector<float>& audio, const SynthSignal& sig, int delay,
                           double settle_s, double edge_s) {
    const double fs = sig.sample_rate;
    const auto frames = yin_track(audio, fs, 60.0, 1000.0, 0.005);
    VibratoStats out;
    const std::size_t n = sig.target.size();
    std::size_t i = 0;
    while (i < n) {
        if (sig.target[i] < 0 || !sig.onset[i]) { ++i; continue; }
        std::size_t j = i + 1;
        while (j < n && sig.target[j] == sig.target[i] && !sig.onset[j]) ++j;
        const double t0 = static_cast<double>(i) / fs + settle_s;
        const double t1 = static_cast<double>(j) / fs - edge_s;
        const int target = sig.target[i];
        std::vector<double> dev;
        for (const auto& f : frames) {
            const double t_in = f.t - static_cast<double>(delay) / fs;
            if (t_in < t0 || t_in > t1 || f.hz <= 0.0) continue;
            dev.push_back(1200.0 * std::log2(f.hz / midi_to_hz(target)));
        }
        if (dev.size() >= 10) {
            double mean = 0.0;
            for (double d : dev) mean += d;
            mean /= dev.size();
            double var = 0.0;
            for (double d : dev) var += (d - mean) * (d - mean);
            out.centre_error_cents += std::abs(mean);
            out.depth_cents += std::sqrt(var / dev.size());
            ++out.segments;
        }
        i = j;
    }
    if (out.segments > 0) {
        out.centre_error_cents /= out.segments;
        out.depth_cents /= out.segments;
    }
    return out;
}

}  // namespace tune
