// Track-mode pitch shifter (spec §6.4): period-synchronous resampling.
//
// Input goes into a ring buffer. A read head moves through it at `ratio`
// samples per output sample (ratio > 1 raises pitch). The delay between the
// newest input sample and the read head therefore drifts; when it leaves the
// window [min_delay, min_delay + period], the read head jumps by about one
// input pitch period (refined by a short correlation search) and the two
// heads are crossfaded. Jumping by a whole period keeps the waveform
// continuous for periodic (voiced) input.
//
// Timing:
//   * reported latency = min_delay (the delay at rest, ratio 1);
//   * while correcting, the delay wanders within [min_delay, min_delay + P],
//     where P is the input period. This wander is inherent to any
//     time-domain shifter and is NOT host-compensated latency;
//   * when the input is unvoiced the head returns to min_delay with a
//     crossfade, so consonants stay aligned.
//
// Real-time safe: allocation only in prepare().
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace tune {

struct ShifterConfig {
    int min_delay = 96;       // samples; reported latency
    int crossfade = 64;       // samples
    int max_period = 700;     // samples (lowest f0 the shifter must handle)
    double max_ratio = 2.0;   // highest pitch factor (+12 st)
    double search = 0.03;     // +/- fraction of the period searched for the best jump
    int match_window = 32;    // samples compared when refining a jump
    int rest_after = 960;     // samples of continuous "unvoiced" before the head
                              // returns to the rest delay (20 ms at 48 kHz)
};

class SpliceShifter {
public:
    void prepare(const ShifterConfig& cfg) {
        cfg_ = cfg;
        // Crossfading while reading faster than real time eats into the delay:
        // the old head must never overtake the write head.
        const int need = static_cast<int>(std::ceil((cfg.max_ratio - 1.0) * cfg.crossfade)) + 4;
        if (cfg_.min_delay < need) cfg_.min_delay = need;
        std::size_t size = 1;
        const std::size_t want = static_cast<std::size_t>(
            cfg_.min_delay + 2 * cfg_.max_period + 2 * cfg_.crossfade + cfg_.match_window + 16);
        while (size < want) size <<= 1;
        buf_.assign(size, 0.0f);
        match_.assign(size, 0.0f);
        mask_ = size - 1;
        reset();
    }

    void reset() {
        std::fill(buf_.begin(), buf_.end(), 0.0f);
        std::fill(match_.begin(), match_.end(), 0.0f);
        w_ = 0;
        r_ = -static_cast<double>(cfg_.min_delay);
        fading_ = false;
        fade_pos_ = 0;
        jumps_ = 0;
        unvoiced_run_ = 0;
        last_delay_ = cfg_.min_delay;
    }

    int latency() const { return cfg_.min_delay; }
    std::int64_t jump_count() const { return jumps_; }
    // Delay of the (primary) read head behind the newest input sample at the
    // last processed sample.
    double current_delay() const { return last_delay_; }

    // ratio: pitch factor for this sample. period: input pitch period in
    // samples, or 0 if unvoiced.
    float process(float x, double ratio, double period) { return process(x, x, ratio, period); }

    // As above, but splice points are chosen on `match` (e.g. the original
    // waveform) while `x` (e.g. the formant-whitened residual) is what gets
    // shifted. A flat, noise-like residual makes poor splice decisions
    // (spike v0.2 finding), the waveform does not.
    float process(float x, float match, double ratio, double period) {
        buf_[static_cast<std::size_t>(w_) & mask_] = x;
        match_[static_cast<std::size_t>(w_) & mask_] = match;
        const std::int64_t newest = w_;
        ++w_;

        unvoiced_run_ = period > 0.0 ? 0 : unvoiced_run_ + 1;
        if (!fading_) maybe_jump(newest, ratio, period);
        last_delay_ = static_cast<double>(newest) - r_;

        float y;
        if (fading_) {
            const float g = (fade_pos_ + 0.5f) / static_cast<float>(cfg_.crossfade);
            y = (1.0f - g) * read(r_old_) + g * read(r_);
            r_old_ += ratio;
            if (++fade_pos_ >= cfg_.crossfade) fading_ = false;
        } else {
            y = read(r_);
        }
        r_ += ratio;
        return y;
    }

private:
    void maybe_jump(std::int64_t newest, double ratio, double period) {
        const double d = static_cast<double>(newest) - r_;
        const double dmin = cfg_.min_delay;
        if (period <= 0.0) {
            // Unvoiced for long enough: return to the rest delay (exact integer
            // position) so timing matches the reported latency again. Waiting
            // avoids large unaligned jumps on short voicing dropouts.
            if (unvoiced_run_ >= cfg_.rest_after && std::abs(d - dmin) > 0.5)
                start_fade(static_cast<double>(newest) - dmin);
            return;
        }
        const double p = std::min(period, static_cast<double>(cfg_.max_period));
        if (ratio > 1.0 && d < dmin + 1.0) {
            start_fade(r_ - refine(r_, p, -1));
        } else if (ratio < 1.0 && d > dmin + p) {
            start_fade(r_ + refine(r_, p, +1));
        }
    }

    // Best jump distance near `p` (in samples), comparing the samples just
    // behind the current head with those behind the candidate position.
    double refine(double r, double p, int dir) const {
        const int lo = static_cast<int>(std::floor(p * (1.0 - cfg_.search)));
        const int hi = static_cast<int>(std::ceil(p * (1.0 + cfg_.search)));
        const int win = cfg_.match_window;
        const std::int64_t base = static_cast<std::int64_t>(std::floor(r));
        double best_score = -1e30;
        int best = static_cast<int>(std::lround(p));
        for (int lag = std::max(lo, 1); lag <= hi; ++lag) {
            const std::int64_t other = base + dir * lag;
            double xy = 0.0, xx = 0.0, yy = 0.0;
            for (int k = 1; k <= win; ++k) {
                const double a = match_at(base - k);
                const double b = match_at(other - k);
                xy += a * b;
                xx += a * a;
                yy += b * b;
            }
            const double score = xy / std::sqrt(xx * yy + 1e-20);
            if (score > best_score) {
                best_score = score;
                best = lag;
            }
        }
        return static_cast<double>(best);
    }

    void start_fade(double new_r) {
        r_old_ = r_;
        r_ = new_r;
        fading_ = true;
        fade_pos_ = 0;
        ++jumps_;
    }

    float at(std::int64_t i) const { return buf_[static_cast<std::size_t>(i) & mask_]; }
    float match_at(std::int64_t i) const { return match_[static_cast<std::size_t>(i) & mask_]; }

    // Cubic Hermite (Catmull-Rom) interpolation; exact at integer positions.
    float read(double pos) const {
        const double fl = std::floor(pos);
        const std::int64_t i = static_cast<std::int64_t>(fl);
        const float t = static_cast<float>(pos - fl);
        const float xm1 = at(i - 1), x0 = at(i), x1 = at(i + 1), x2 = at(i + 2);
        if (t == 0.0f) return x0;
        const float c1 = 0.5f * (x1 - xm1);
        const float c2 = xm1 - 2.5f * x0 + 2.0f * x1 - 0.5f * x2;
        const float c3 = 0.5f * (x2 - xm1) + 1.5f * (x0 - x1);
        return ((c3 * t + c2) * t + c1) * t + x0;
    }

    ShifterConfig cfg_{};
    std::vector<float> buf_, match_;
    std::size_t mask_ = 0;
    std::int64_t w_ = 0;
    double r_ = 0.0, r_old_ = 0.0;
    bool fading_ = false;
    int fade_pos_ = 0;
    std::int64_t jumps_ = 0;
    int unvoiced_run_ = 0;
    double last_delay_ = 0.0;
};

}  // namespace tune
