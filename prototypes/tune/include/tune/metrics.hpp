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

}  // namespace tune
