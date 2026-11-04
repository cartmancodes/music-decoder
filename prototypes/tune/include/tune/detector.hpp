// Causal per-sample pitch detector for the tuner (spec §6.1): cycfi/q's
// bitstream-autocorrelation pitch_detector on a band-limited copy of the
// input, plus a voicing gate (level + periodicity + hold).
//
// Allocation happens only in prepare() (q's detector allocates its bit
// buffers when constructed); process() must not allocate.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <vector>

#include <q/pitch/pitch_detector.hpp>
#include <q/support/literals.hpp>

namespace tune {

struct DetectorConfig {
    double min_hz = 70.0;
    double max_hz = 800.0;
    double gate_db = -50.0;          // envelope below this = unvoiced
    double min_periodicity = 0.80;   // q periodicity below this = unvoiced
    double hold_ms = 12.0;           // minimum hold after the last good estimate
    double hold_periods = 3.5;       // ...and at least this many periods: q reports
                                     // roughly every 2 periods, so a shorter hold
                                     // makes voicing flicker inside steady notes
    // Period tracker between q's estimates (q updates about every 2 periods
    // and median-filters, which lags vibrato). Every `track_hop_ms`, search a
    // normalised autocorrelation over one period, +/- `track_range` around the
    // last period. q's estimate re-anchors the tracker when they disagree by
    // more than `reanchor` (note changes, octave safety).
    bool track = true;
    double track_hop_ms = 1.33;
    double track_range = 0.04;
    double reanchor = 0.06;
    double track_min_nccf = 0.6;
    double track_max_window_ms = 5.33;  // correlation window cap (256 samples at 48 kHz);
                                        // a duration, so 96 kHz keeps enough of a low period
    double track_hop_periods = 0.25; // update at least every quarter period
    // Fast voicing end (onset fix, spike v0.1): if the tracker's best
    // correlation stays below `unvoice_nccf` for `drop_hops` consecutive
    // updates (consonants, noise, silence), end voicing and reset q so the
    // next note is acquired fresh instead of inheriting a stale period.
    double unvoice_nccf = 0.5;
    int drop_hops = 2;
    // Fast acquisition (onset fix, spike v0.1): while unvoiced but above the
    // gate, run a YIN search (cumulative-mean-normalised difference) over all
    // lags the history allows, every tracker hop. Two consecutive candidates
    // within `acquire_agree` start voicing. q's own proposals from the
    // unvoiced state must pass an NCCF check (`confirm_nccf`), which stops q
    // voicing on consonant noise.
    bool acquire = true;
    double acquire_window_ms = 4.0;
    double acquire_threshold = 0.12;
    double acquire_agree = 0.03;
    double confirm_nccf = 0.8;
};

// RBJ biquad, transposed direct form II (double precision state).
class Biquad {
public:
    void set_highpass(double fs, double f0, double q) { design(fs, f0, q, true); }
    void set_lowpass(double fs, double f0, double q) { design(fs, f0, q, false); }
    void reset() { z1_ = z2_ = 0.0; }
    double process(double x) {
        const double y = b0_ * x + z1_;
        z1_ = b1_ * x - a1_ * y + z2_;
        z2_ = b2_ * x - a2_ * y;
        return y;
    }

private:
    void design(double fs, double f0, double q, bool hp) {
        const double w = 2.0 * 3.14159265358979323846 * f0 / fs;
        const double cw = std::cos(w), sw = std::sin(w);
        const double alpha = sw / (2.0 * q);
        const double a0 = 1.0 + alpha;
        if (hp) {
            b0_ = (1.0 + cw) / 2.0 / a0;
            b1_ = -(1.0 + cw) / a0;
            b2_ = b0_;
        } else {
            b0_ = (1.0 - cw) / 2.0 / a0;
            b1_ = (1.0 - cw) / a0;
            b2_ = b0_;
        }
        a1_ = -2.0 * cw / a0;
        a2_ = (1.0 - alpha) / a0;
    }
    double b0_ = 1, b1_ = 0, b2_ = 0, a1_ = 0, a2_ = 0;
    double z1_ = 0, z2_ = 0;
};

class Detector {
public:
    void prepare(double fs, const DetectorConfig& cfg) {
        using namespace cycfi::q::literals;
        fs_ = fs;
        cfg_ = cfg;
        pd_.emplace(cycfi::q::frequency(cfg.min_hz), cycfi::q::frequency(cfg.max_hz),
                    static_cast<float>(fs), -45_dB);
        hp_.set_highpass(fs, cfg.min_hz * 0.7, 0.707);
        lp_.set_lowpass(fs, std::min(cfg.max_hz * 3.0, 0.45 * fs), 0.707);
        env_a_ = std::exp(-1.0 / (0.010 * fs));  // 10 ms release envelope
        gate_lin_ = std::pow(10.0, cfg.gate_db / 20.0);
        hold_samples_ = static_cast<int>(cfg.hold_ms * 1e-3 * fs);
        max_period_ = static_cast<int>(std::ceil(fs / cfg.min_hz)) + 2;
        min_period_ = std::max(2, static_cast<int>(std::floor(fs / cfg.max_hz)));
        acq_window_ = std::max(32, static_cast<int>(std::lround(cfg.acquire_window_ms * 1e-3 * fs)));
        // Acquisition runs on a decimated copy (the detection path is already
        // low-passed at <= 2.4 kHz; a 4-sample average adds anti-aliasing),
        // ~16x cheaper than full rate. Candidates are refined at full rate.
        dec_ = std::max(1, static_cast<int>(std::lround(fs / 12000.0)));
        dec_max_period_ = max_period_ / dec_ + 2;
        dec_min_period_ = std::max(2, min_period_ / dec_);
        dec_window_ = std::max(16, acq_window_ / dec_);
        std::size_t dsize = 1;
        while (dsize < static_cast<std::size_t>(dec_max_period_ + dec_window_ + 8)) dsize <<= 1;
        dhist_.assign(dsize, 0.0f);
        dhist_mask_ = dsize - 1;
        cmnd_.assign(static_cast<std::size_t>(dec_max_period_) + 2, 0.0);
        std::size_t size = 1;
        while (size < static_cast<std::size_t>(3 * max_period_ + 16 +
                                               static_cast<int>(cfg.acquire_window_ms * 1e-3 * fs)))
            size <<= 1;
        hist_.assign(size, 0.0f);
        hist_mask_ = size - 1;
        track_hop_ = std::max(1, static_cast<int>(std::lround(cfg.track_hop_ms * 1e-3 * fs)));
        track_max_window_ = std::max(64, static_cast<int>(std::lround(cfg.track_max_window_ms * 1e-3 * fs)));
        reset();
    }

    void reset() {
        hp_.reset();
        lp_.reset();
        env_ = 0.0;
        hz_ = 0.0;
        voiced_ = false;
        hold_ = 0;
        if (pd_) pd_->reset();
        std::fill(hist_.begin(), hist_.end(), 0.0f);
        hn_ = 0;
        period_ = 0.0;
        hop_count_ = 0;
        fails_ = 0;
        acq_count_ = 0;
        acq_last_ = 0.0;
        acq_hop_ = 0;
        std::fill(dhist_.begin(), dhist_.end(), 0.0f);
        dn_ = 0;
        dacc_ = 0.0;
        dphase_ = 0;
    }

    // Feed one input sample. Returns the current f0 estimate in Hz, or 0 when
    // unvoiced.
    double process(float x) {
        const double band = lp_.process(hp_.process(x));
        hist_[static_cast<std::size_t>(hn_) & hist_mask_] = static_cast<float>(band);
        ++hn_;
        dacc_ += band;
        if (++dphase_ >= dec_) {
            dhist_[static_cast<std::size_t>(dn_) & dhist_mask_] = static_cast<float>(dacc_ / dec_);
            ++dn_;
            dacc_ = 0.0;
            dphase_ = 0;
        }
        const double ax = std::abs(band);
        env_ = ax > env_ ? ax : ax + env_a_ * (env_ - ax);

        const bool ready = (*pd_)(static_cast<float>(band));
        if (ready) {
            const double f = pd_->get_frequency();
            const double per = pd_->periodicity();
            bool accept = f >= cfg_.min_hz * 0.97 && f <= cfg_.max_hz * 1.03 &&
                          per >= cfg_.min_periodicity && env_ >= gate_lin_;
            double q_refined = f > 0.0 ? fs_ / f : 0.0;
            double q_r = 1.0;
            if (accept && cfg_.track) q_r = search(fs_ / f, q_refined);
            if (accept && !voiced_ && cfg_.track) {
                // From unvoiced, q must be confirmed by the signal itself.
                accept = q_r >= cfg_.confirm_nccf;
            }
            if (accept) {
                const double p = fs_ / f;
                if (!cfg_.track || period_ <= 0.0) {
                    period_ = cfg_.track ? q_refined : p;
                } else if (std::abs(p - period_) > cfg_.reanchor * period_) {
                    // Re-anchor only if q's period explains the signal at
                    // least as well as the current track (note change), not
                    // on a stray q estimate right after a reset.
                    double cur_refined = period_;
                    const double cur_r = search(period_, cur_refined);
                    if (q_r >= cfg_.confirm_nccf && q_r >= cur_r - 0.02) period_ = q_refined;
                }
                hz_ = fs_ / period_;
                hold_ = std::max(hold_samples_,
                                 static_cast<int>(cfg_.hold_periods * fs_ / f));
                voiced_ = true;
            }
        }
        if (cfg_.acquire && !voiced_ && env_ >= gate_lin_ && ++acq_hop_ >= track_hop_) {
            acq_hop_ = 0;
            double cand = acquire_period();
            if (cand > 0.0 && dec_ > 1) {
                // Refine the coarse (decimated) period at full rate.
                double refined = cand;
                cand = search(cand, refined) >= cfg_.confirm_nccf ? refined : 0.0;
            }
            if (cand > 0.0 && acq_last_ > 0.0 &&
                std::abs(cand - acq_last_) <= cfg_.acquire_agree * acq_last_) {
                if (++acq_count_ >= 1) {
                    period_ = cand;
                    hz_ = fs_ / period_;
                    voiced_ = true;
                    fails_ = 0;
                    hop_count_ = 0;
                    hold_ = std::max(hold_samples_, static_cast<int>(cfg_.hold_periods * period_));
                    acq_count_ = 0;
                }
            } else {
                acq_count_ = 0;
            }
            acq_last_ = cand;
        }
        if (voiced_) acq_last_ = 0.0;
        if (cfg_.track && voiced_ && period_ > 0.0 &&
            ++hop_count_ >= std::max(track_hop_, static_cast<int>(cfg_.track_hop_periods * period_))) {
            hop_count_ = 0;
            const double r = refine_period();
            hz_ = fs_ / period_;
            if (r >= cfg_.unvoice_nccf) {
                fails_ = 0;
                // A periodic track keeps the voice alive between q updates.
                hold_ = std::max(hold_, std::max(hold_samples_,
                                 static_cast<int>(cfg_.hold_periods * period_)));
            } else if (++fails_ >= cfg_.drop_hops) {
                hold_ = 0;  // ends voicing below
            }
        }
        if (env_ < gate_lin_) hold_ = 0;
        if (hold_ > 0) {
            --hold_;
        } else if (voiced_) {
            voiced_ = false;
            hz_ = 0.0;
            period_ = 0.0;
            fails_ = 0;
            pd_->reset();
        }
        return voiced_ ? hz_ : 0.0;
    }

    bool voiced() const { return voiced_; }
    double frequency() const { return voiced_ ? hz_ : 0.0; }

private:
    float h(std::int64_t i) const { return hist_[static_cast<std::size_t>(i) & hist_mask_]; }

    float dh(std::int64_t i) const { return dhist_[static_cast<std::size_t>(i) & dhist_mask_]; }

    // YIN on the most recent (decimated) samples. Returns the period in
    // full-rate samples, or 0 if nothing passes the threshold (noise,
    // silence, onset not yet fully in the window).
    double acquire_period() {
        const std::int64_t n = dn_ - 1;
        const int w = dec_window_;
        const int tmax = dec_max_period_;
        if (n - w - tmax - 1 < 0) return 0.0;
        double running = 0.0;
        cmnd_[0] = 1.0;
        for (int tau = 1; tau <= tmax + 1; ++tau) {
            double d = 0.0;
            for (int k = 0; k < w; ++k) {
                const double diff = double(dh(n - k)) - dh(n - k - tau);
                d += diff * diff;
            }
            running += d;
            cmnd_[tau] = running > 0.0 ? d * tau / running : 1.0;
        }
        for (int tau = dec_min_period_; tau <= tmax; ++tau) {
            if (cmnd_[tau] < cfg_.acquire_threshold) {
                while (tau + 1 <= tmax && cmnd_[tau + 1] < cmnd_[tau]) ++tau;
                const double a = cmnd_[tau - 1], b = cmnd_[tau], c = cmnd_[tau + 1];
                const double denom = a - 2 * b + c;
                const double shift = std::abs(denom) > 1e-12 ? std::clamp(0.5 * (a - c) / denom, -0.5, 0.5) : 0.0;
                return (tau + shift) * dec_;
            }
        }
        return 0.0;
    }

    // Normalised autocorrelation over one period (window capped), searched
    // +/- track_range around `center`. Returns the best correlation (1.0 if
    // there is not enough history yet, so callers don't count a failure) and
    // writes the sub-sample refined period. Does not modify state.
    double search(double center, double& refined) const {
        refined = center;
        const std::int64_t n = hn_ - 1;
        // A window shorter than one period still peaks at the period lag for
        // periodic input; capping it bounds the cost on low voices.
        const int w = std::clamp(static_cast<int>(std::lround(center)), 64, track_max_window_);
        const int lo = std::max(2, static_cast<int>(std::floor(center * (1.0 - cfg_.track_range))));
        const int hi = std::min(max_period_, static_cast<int>(std::ceil(center * (1.0 + cfg_.track_range))));
        if (n - w - hi - 1 < 0 || hi - lo < 2) return 1.0;
        double xx = 0.0;
        for (int k = 0; k < w; ++k) xx += double(h(n - k)) * h(n - k);
        double best = -2.0, prev = -2.0, before_best = -2.0, after_best = -2.0;
        int best_lag = -1;
        for (int lag = lo - 1; lag <= hi + 1; ++lag) {
            double xy = 0.0, yy = 0.0;
            for (int k = 0; k < w; ++k) {
                const double a = h(n - k), b = h(n - k - lag);
                xy += a * b;
                yy += b * b;
            }
            const double r = xy / std::sqrt(xx * yy + 1e-20);
            if (best_lag == lag - 1) after_best = r;
            if (lag >= lo && lag <= hi && r > best) {
                best = r;
                best_lag = lag;
                before_best = prev;
                after_best = -2.0;
            }
            prev = r;
        }
        if (best_lag < 0) return best;
        double shift = 0.0;
        const double denom = before_best - 2.0 * best + after_best;
        if (before_best > -2.0 && after_best > -2.0 && std::abs(denom) > 1e-12)
            shift = std::clamp(0.5 * (before_best - after_best) / denom, -0.5, 0.5);
        refined = best_lag + shift;
        return best;
    }

    // Track: refine period_ in place when the peak is strong enough.
    double refine_period() {
        double refined = period_;
        const double r = search(period_, refined);
        if (r >= cfg_.track_min_nccf) period_ = refined;
        return r;
    }

    double fs_ = 48000.0;
    DetectorConfig cfg_{};
    std::optional<cycfi::q::pitch_detector> pd_;
    Biquad hp_, lp_;
    double env_ = 0.0, env_a_ = 0.0, gate_lin_ = 0.0;
    double hz_ = 0.0;
    bool voiced_ = false;
    int hold_ = 0, hold_samples_ = 0;
    // tracker
    std::vector<float> hist_;
    std::size_t hist_mask_ = 0;
    std::int64_t hn_ = 0;
    double period_ = 0.0;
    int max_period_ = 0, track_hop_ = 64, hop_count_ = 0;
    int fails_ = 0;
    int track_max_window_ = 256;
    // acquisition
    std::vector<double> cmnd_;
    int min_period_ = 2, acq_window_ = 192, acq_hop_ = 0, acq_count_ = 0;
    double acq_last_ = 0.0;
    // decimated history for acquisition
    std::vector<float> dhist_;
    std::size_t dhist_mask_ = 0;
    std::int64_t dn_ = 0;
    double dacc_ = 0.0;
    int dec_ = 4, dphase_ = 0, dec_max_period_ = 0, dec_min_period_ = 2, dec_window_ = 48;
};

}  // namespace tune
