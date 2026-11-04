// Objective tuner metrics on synthetic signals (spec §15.1, §15.5).
#pragma once

#include <string>
#include <vector>

#include "tune/synth.hpp"

namespace tune {

struct PitchStats {
    int frames = 0;              // scored frames
    double median_abs_cents = 0;
    double p95_abs_cents = 0;
    double within10_pct = 0;     // % of frames within +/-10 cents of target
    double gross50_pct = 0;      // % of frames off by > 50 cents
    double octave_pct = 0;       // % of frames off by > 1000 cents
};

// Scores `audio` (input or tuner output) against the signal's per-sample
// targets. `delay_samples` aligns output to input; frames within `settle_s`
// after a voiced onset, or within `edge_s` of a segment end, are skipped.
PitchStats score_pitch(const std::vector<float>& audio, const SynthSignal& sig,
                       int delay_samples, double settle_s = 0.060, double edge_s = 0.030);

// Energy above `cut_hz` in voiced, scored regions relative to total energy
// there (dB). Splice clicks raise it because the synthetic voice is
// band-limited to 8 kHz.
double hf_energy_db(const std::vector<float>& audio, const SynthSignal& sig, int delay_samples,
                    double cut_hz = 12000.0);

std::string format_stats(const PitchStats& s);

// Onset settle time (plan T1.2): for each scored voiced segment, the time
// from onset until the output stays within +/-`tol_cents` of the target for
// at least `hold_s`. The YIN judge (2 ms hop) needs a full window inside the
// segment, so half its window is subtracted as judge bias (clamped at 0).
struct SettleStats {
    int segments = 0;       // scored segments
    int unsettled = 0;      // segments that never settled
    double median_ms = 0;   // over settled segments
    double p90_ms = 0;
};
SettleStats score_settle(const std::vector<float>& audio, const SynthSignal& sig,
                         int delay_samples, double tol_cents = 10.0, double hold_s = 0.020);

// Real clips have no ground truth. This scores how close voiced frames are
// to the nearest note of `scale_mask` (bit i = pitch class i): a tuned output
// should sit on scale notes; the input usually does not. Frames with YIN
// aperiodicity above `max_aperiodicity` are skipped.
PitchStats score_against_scale(const std::vector<float>& audio, double fs,
                               unsigned scale_mask, double min_hz = 60.0,
                               double max_hz = 1000.0, double max_aperiodicity = 0.2);

// Formant envelope of a steady harmonic segment (spec FR-TUNE-6).
// Harmonic amplitudes are measured by Hann-windowed projection at k * f0
// over [start, start + len); multiplying by k removes the synthetic voice's
// 1/k tilt, giving envelope samples at k * f0 for 150 Hz..max_hz.
struct EnvelopeFit {
    double lsd_db = 0;     // RMS dB difference vs the reference envelope (both mean-normalised)
    double f1_hz = 0;      // envelope peak in 450-950 Hz (parabolic, log domain)
    double f2_hz = 0;      // envelope peak in 950-1600 Hz
    int harmonics = 0;
};
// `reference(f)` is the expected envelope (linear amplitude) at frequency f.
template <class F>
EnvelopeFit fit_envelope(const std::vector<float>& audio, double fs, std::size_t start,
                         std::size_t len, double f0, F reference, double max_hz = 3500.0);

}  // namespace tune

#include "tune/metrics_envelope.inl"
