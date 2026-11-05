// Track-mode tuner: detector -> note decider -> retune smoother -> splice
// shifter, with LPC formant preservation around the shifter (spec §6.1-6.5)
// and the humanize / flex / vibrato controls (§6.3). Mono, per-sample
// processing; block size has no effect on the output (NFR-6).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "tune/correction.hpp"
#include "tune/detector.hpp"
#include "tune/formant.hpp"
#include "tune/scale.hpp"
#include "tune/splice_shifter.hpp"

namespace tune {

// Formant handling (spec §6.5). Auto preserves formants only when the user
// transposes (|transpose| >= 0.5 st): plain hard-tune corrections are small
// (<= ~50 c, <= 3% formant shift) and the formant chain costs a little pitch
// tightness on vibrato and onsets (spike v0.2), so Auto skips it there.
enum class FormantMode { Auto, Preserve, Follow };

struct TunerParams {
    int tonic = 0;                        // pitch class, C = 0
    ScaleType scale = ScaleType::Major;
    double retune_ms = 0.0;               // 0 = hard-tune
    double amount = 1.0;                  // 0..1
    double hysteresis_cents = 15.0;
    double transpose_semitones = 0.0;
    double max_correction_cents = 600.0;  // clamp (+/- half an octave)
    FormantMode formant_mode = FormantMode::Auto;
    // Spec §6.3 controls (spike v0.3):
    double humanize = 0.0;      // 0..1: retune time grows with how long a note is held,
                                // +humanize * 120 ms after 300 ms (also at retune 0)
    double flex = 0.0;          // 0..1: 0 = off. Otherwise correction fades out when the
                                // voice is far from the target (window 100 c at flex
                                // ~0 down to 15 c at flex 1): scoops and slides pass
    double vibrato = 0.0;       // 0..2: gain on the singer's own vibrato around the
                                // corrected note. 0 = flatten (classic hard-tune),
                                // 1 = keep natural vibrato, 2 = exaggerate
    double throat_semitones = 0.0;  // formant shift on top of the formant mode
                                    // (spec: -100..+100 % = -3..+3 st); + = smaller,
                                    // brighter voice. Engages the formant chain.
};

// Control constants (spec §6.3, [tune] values from spike v0.3).
struct ControlConstants {
    static constexpr double humanize_max_ms = 120.0;
    static constexpr double humanize_full_after_ms = 300.0;
    static constexpr double flex_window_max_cents = 100.0;
    static constexpr double flex_window_min_cents = 15.0;
    static constexpr double centre_cutoff_hz = 2.0;  // centre / vibrato split
    static constexpr int control_block = 32;          // samples between humanize / throat updates
    static constexpr double throat_glide_ms = 20.0;   // throat parameter smoothing
    static constexpr double throat_max_semitones = 6.0;  // UI range is +/-3 st (spec); +/-6 accepted
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
    void prepare(double sample_rate, const DetectorConfig& det = {},
                 const FormantConfig& formant = {}) {
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

        formant_.prepare(fs_, formant);
        smoother_.prepare(fs_);
        release_.prepare(fs_, 5.0);
        centre_.prepare(fs_, ControlConstants::centre_cutoff_hz);
        set_params(params_);
        reset();
    }

    void reset() {
        detector_.reset();
        decider_.reset();
        smoother_.reset(0.0);
        release_.reset(0.0);
        shifter_.reset();
        formant_.reset();
        centre_.reset(0.0);
        throat_s_ = params_.throat_semitones;
        was_voiced_ = false;
        note_samples_ = 0;
        last_target_ = -1;
        applied_retune_ms_ = -1.0;
        n_ = 0;
        last_ = {};
    }

    // Real-time safe (no allocation). Takes effect from the next sample.
    void set_params(const TunerParams& p) {
        params_ = p;
        decider_.set_scale(Scale::make(p.tonic, p.scale));
        decider_.set_hysteresis_cents(p.hysteresis_cents);
        smoother_.set_retune_ms(p.retune_ms);
        applied_retune_ms_ = p.retune_ms;
    }

    const TunerParams& params() const { return params_; }
    bool preserving() const {
        return params_.formant_mode == FormantMode::Preserve ||
               (params_.formant_mode == FormantMode::Auto && std::abs(params_.transpose_semitones) >= 0.5);
    }
    // The formant chain runs when preserving or when Throat is (or is still
    // gliding) away from 0.
    bool chain_active() const {
        return preserving() || params_.throat_semitones != 0.0 || std::abs(throat_s_) > 1e-3;
    }
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
            const int t_note = static_cast<int>(target);
            // A new note (voicing onset or target change) restarts the
            // humanize clock and the centre filter, so the vibrato split
            // never sees the jump between notes.
            if (!was_voiced_ || t_note != last_target_) {
                note_samples_ = 0;
                centre_.reset(midi_in);
            }
            last_target_ = t_note;
            const double centre = centre_.process(midi_in);
            const double vib = midi_in - centre;

            // Flex: weight by the centre's distance from the target.
            double weight = 1.0;
            if (params_.flex > 0.0) {
                const double f = std::clamp(params_.flex, 0.0, 1.0);
                const double window = ControlConstants::flex_window_max_cents +
                                      (ControlConstants::flex_window_min_cents -
                                       ControlConstants::flex_window_max_cents) * f;
                const double d = std::abs(target - centre) * 100.0;
                weight = d <= 0.5 * window ? 1.0
                       : d >= window      ? 0.0
                                          : (window - d) / (0.5 * window);
            }
            // Output pitch -> target + vibrato_gain * vib (at full amount and
            // retune 0); vibrato 0 gives the classic flat hard-tune.
            double want = ((target - midi_in) + params_.vibrato * vib) * 100.0 *
                          params_.amount * weight;
            want = std::clamp(want, -params_.max_correction_cents, params_.max_correction_cents);

            // Humanize: longer-held notes get a slower retune.
            if (params_.humanize > 0.0 && note_samples_ % ControlConstants::control_block == 0) {
                const double held_ms = 1000.0 * static_cast<double>(note_samples_) / fs_;
                const double grow = std::min(held_ms / ControlConstants::humanize_full_after_ms, 1.0);
                const double rt = params_.retune_ms +
                                  std::clamp(params_.humanize, 0.0, 1.0) *
                                      ControlConstants::humanize_max_ms * grow;
                if (std::abs(rt - applied_retune_ms_) > 0.25) {
                    smoother_.set_retune_ms(rt);
                    applied_retune_ms_ = rt;
                }
            }
            ++note_samples_;
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
        // The analysis always runs (so toggling Preserve has history); the
        // whitened signal is only used when preserving formants.
        if (n_ % ControlConstants::control_block == 0) {
            // Throat glide, then the re-colour envelope scale: Preserve keeps
            // the envelope (scale = throat); Auto / Follow let it move with the
            // pitch shift as plain resampling would (scale = ratio x throat).
            const double target_st = std::clamp(params_.throat_semitones,
                                                -ControlConstants::throat_max_semitones,
                                                ControlConstants::throat_max_semitones);
            const double a = std::exp(-ControlConstants::control_block /
                                      (ControlConstants::throat_glide_ms * 1e-3 * fs_));
            throat_s_ = target_st + a * (throat_s_ - target_st);
            if (std::abs(throat_s_ - target_st) < 1e-4) throat_s_ = target_st;
            const double follow = preserving() ? 1.0 : ratio;
            formant_.set_recolour_scale(follow * std::exp2(throat_s_ / 12.0));
        }
        const float e = formant_.whiten(x);
        float y;
        if (chain_active()) {
            const float ey = shifter_.process(e, x, ratio, period);
            // Re-apply the envelope of the input moment being read.
            y = formant_.recolour(ey, static_cast<double>(n_) - shifter_.current_delay());
        } else {
            y = shifter_.process(x, ratio, period);
        }

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
    ControlLowpass centre_;
    SpliceShifter shifter_;
    FormantChain formant_;
    bool was_voiced_ = false;
    std::int64_t note_samples_ = 0;
    int last_target_ = -1;
    double applied_retune_ms_ = -1.0;
    double throat_s_ = 0.0;
    std::int64_t n_ = 0;
    TuneFrame last_{};
};

}  // namespace tune
