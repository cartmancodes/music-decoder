// Track-mode tuner: detector -> note decider -> retune smoother -> splice
// shifter (spec §6.1-6.4). Mono, per-sample processing; block size has no
// effect on the output (NFR-6). Formant preservation (§6.5), humanize, flex
// and vibrato (§6.3) are later spike steps.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "tune/correction.hpp"
#include "tune/detector.hpp"
#include "tune/scale.hpp"
#include "tune/splice_shifter.hpp"

namespace tune {

struct TunerParams {
    int tonic = 0;                        // pitch class, C = 0
    ScaleType scale = ScaleType::Major;
    double retune_ms = 0.0;               // 0 = hard-tune
    double amount = 1.0;                  // 0..1
    double hysteresis_cents = 15.0;
    double transpose_semitones = 0.0;
    double max_correction_cents = 600.0;  // clamp (+/- half an octave)
};

// One analysis frame for traces / UI (spec §11.3).
struct TuneFrame {
    double t = 0.0;          // seconds
    double f0_in = 0.0;      // Hz, 0 = unvoiced
    double target_midi = 0;  // 0 when unvoiced
    double correction = 0;   // cents applied
    double delay = 0;        // samples, shifter read-head delay
};

class Tuner {
public:
    // Allocates. Call before processing and whenever the sample rate changes.
    void prepare(double sample_rate, const DetectorConfig& det = {}) {
        fs_ = sample_rate;
        det_cfg_ = det;
        detector_.prepare(fs_, det);

        ShifterConfig sc;
        sc.min_delay = static_cast<int>(std::lround(0.0020 * fs_));   // 2.0 ms
        sc.crossfade = static_cast<int>(std::lround(0.00133 * fs_));  // 1.33 ms
        sc.max_period = static_cast<int>(std::ceil(fs_ / det.min_hz)) + 4;
        sc.match_window = static_cast<int>(std::lround(0.00067 * fs_));
        sc.rest_after = static_cast<int>(std::lround(0.020 * fs_));
        shifter_.prepare(sc);

        smoother_.prepare(fs_);
        release_.prepare(fs_, 5.0);
        set_params(params_);
        reset();
    }

    void reset() {
        detector_.reset();
        decider_.reset();
        smoother_.reset(0.0);
        release_.reset(0.0);
        shifter_.reset();
        was_voiced_ = false;
        n_ = 0;
        last_ = {};
    }

    // Real-time safe (no allocation). Takes effect from the next sample.
    void set_params(const TunerParams& p) {
        params_ = p;
        decider_.set_scale(Scale::make(p.tonic, p.scale));
        decider_.set_hysteresis_cents(p.hysteresis_cents);
        smoother_.set_retune_ms(p.retune_ms);
    }

    const TunerParams& params() const { return params_; }
    int latency_samples() const { return shifter_.latency(); }
    std::int64_t jump_count() const { return shifter_.jump_count(); }
    const TuneFrame& last_frame() const { return last_; }

    float process_sample(float x) {
        const double f0 = detector_.process(x);
        double correction = 0.0;
        double period = 0.0;
        double target = 0.0;
        if (f0 > 0.0) {
            const double midi_in = hz_to_midi(f0);
            target = decider_.decide(midi_in);
            double want = (target - midi_in) * 100.0 * params_.amount;
            want = std::clamp(want, -params_.max_correction_cents, params_.max_correction_cents);
            if (!was_voiced_) smoother_.reset(want);  // engage on the first voiced sample
            correction = smoother_.process(want);
            release_.reset(correction);
            period = fs_ / f0;
        } else {
            correction = release_.process(0.0);
            decider_.reset();
        }
        was_voiced_ = f0 > 0.0;
        const double cents = correction + params_.transpose_semitones * 100.0;
        // Unvoiced: ratio exactly 1 so the shifter can return to rest delay.
        const double ratio = period > 0.0 ? std::exp2(cents / 1200.0) : 1.0;
        const float y = shifter_.process(x, ratio, period);

        last_.t = static_cast<double>(n_) / fs_;
        last_.f0_in = f0;
        last_.target_midi = target;
        last_.correction = period > 0.0 ? cents : 0.0;
        last_.delay = shifter_.current_delay();
        ++n_;
        return y;
    }

    void process(const float* in, float* out, std::size_t n) {
        for (std::size_t i = 0; i < n; ++i) out[i] = process_sample(in[i]);
    }

private:
    double fs_ = 48000.0;
    DetectorConfig det_cfg_{};
    TunerParams params_{};
    Detector detector_;
    NoteDecider decider_;
    RetuneSmoother smoother_;
    OnePole release_;
    SpliceShifter shifter_;
    bool was_voiced_ = false;
    std::int64_t n_ = 0;
    TuneFrame last_{};
};

}  // namespace tune
