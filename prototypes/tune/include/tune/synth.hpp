// Synthetic "voice" test signals with known pitch, for objective tuner
// metrics (spec §15.1, §15.5). Additive harmonics shaped by a vowel formant
// envelope, plus sibilant noise for unvoiced segments. Not real-time code.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tune {

// Formant envelope model of the synthetic voice.
//  Gaussian:  three Gaussian bumps over a -30 dB floor (deep valleys; hard
//             for all-pole / LPC models - a stress test).
//  Resonator: cascade of five second-order formant resonators, the classic
//             vowel-synthesizer model (close to real vowels).
enum class VoiceModel { Gaussian, Resonator };

struct Segment {
    double dur_s = 0.2;
    double midi_start = 57.0;  // pitch at segment start (fractional MIDI)
    double midi_end = 57.0;    // linear glide to this
    double vib_cents = 0.0;    // vibrato depth (peak)
    double vib_hz = 5.5;
    bool voiced = true;
    int target = -1;           // expected output note (MIDI) or -1 = don't score
};

struct SynthSignal {
    double sample_rate = 48000.0;
    std::vector<float> audio;
    std::vector<double> f0;        // true input f0 per sample (0 = unvoiced)
    std::vector<int> target;       // expected output note per sample (-1 = don't score)
    std::vector<std::uint8_t> onset;  // 1 at the first sample of each voiced segment
};

SynthSignal synth_voice(const std::vector<Segment>& segs, double fs, std::uint32_t seed = 1,
                        VoiceModel model = VoiceModel::Gaussian);

// The synthetic voice's formant envelope (linear amplitude, before the 1/k
// harmonic tilt). Ground truth for formant-preservation metrics.
double vowel_envelope(double f, VoiceModel model = VoiceModel::Gaussian);

struct Scenario {
    std::string name;
    std::string description;
    int tonic = 0;
    int scale = 0;  // tune::ScaleType as int
    std::vector<Segment> segments;
};

// Built-in scenarios (all in A natural minor unless noted).
std::vector<Scenario> builtin_scenarios();

}  // namespace tune
