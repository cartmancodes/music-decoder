#include "tune/synth.hpp"

#include <algorithm>
#include <cmath>
#include <random>

#include "tune/scale.hpp"

namespace tune {
namespace {

constexpr double kPi = 3.14159265358979323846;

// Vowel /a/-like envelope: three Gaussian formant bumps (the 1/k harmonic
// tilt is applied separately in synth_voice).
double formant_gain(double f) {
    struct F { double fc, bw, amp; };
    static constexpr F fs[3] = {{730.0, 110.0, 1.0}, {1090.0, 130.0, 0.6}, {2440.0, 200.0, 0.35}};
    double g = 0.03;
    for (const auto& x : fs) {
        const double z = (f - x.fc) / x.bw;
        g += x.amp * std::exp(-0.5 * z * z);
    }
    return g;
}

// Second-order band-pass (RBJ, constant peak gain) for sibilant noise.
struct Bandpass {
    double b0, b2, a1, a2, z1 = 0, z2 = 0;
    Bandpass(double fs, double f0, double q) {
        const double w = 2 * kPi * f0 / fs, alpha = std::sin(w) / (2 * q), a0 = 1 + alpha;
        b0 = alpha / a0;
        b2 = -alpha / a0;
        a1 = -2 * std::cos(w) / a0;
        a2 = (1 - alpha) / a0;
    }
    double process(double x) {
        const double y = b0 * x + z1;
        z1 = -a1 * y + z2;
        z2 = b2 * x - a2 * y;
        return y;
    }
};

// Cascade of analog two-pole resonators, unity gain at DC (/a/: F1-F5).
double resonator_gain(double f) {
    struct R { double fc, bw; };
    static constexpr R rs[5] = {{730, 90}, {1090, 110}, {2440, 170}, {3300, 250}, {3750, 300}};
    double g = 1.0;
    for (const auto& r : rs) {
        const double q = r.fc / r.bw;
        const double x = f / r.fc;
        g *= 1.0 / std::sqrt((1.0 - x * x) * (1.0 - x * x) + (x / q) * (x / q));
    }
    return g * 0.05;  // keep the overall level similar to the Gaussian model
}

}  // namespace

double vowel_envelope(double f, VoiceModel model) {
    return model == VoiceModel::Resonator ? resonator_gain(f) : formant_gain(f);
}

SynthSignal synth_voice(const std::vector<Segment>& segs, double fs, std::uint32_t seed,
                        VoiceModel model) {
    SynthSignal s;
    s.sample_rate = fs;
    std::mt19937 rng(seed);
    std::normal_distribution<double> noise(0.0, 1.0);
    Bandpass sib(fs, 6500.0, 1.2);
    double phase = 0.0;  // fundamental phase, radians
    const double max_harm_hz = 8000.0;

    for (std::size_t si = 0; si < segs.size(); ++si) {
        const auto& seg = segs[si];
        const bool next_legato = si + 1 < segs.size() && segs[si + 1].legato && segs[si + 1].voiced;
        const std::size_t n = static_cast<std::size_t>(std::lround(seg.dur_s * fs));
        const std::size_t attack = static_cast<std::size_t>(0.010 * fs);
        const std::size_t release = static_cast<std::size_t>(0.015 * fs);
        for (std::size_t i = 0; i < n; ++i) {
            const double u = n > 1 ? static_cast<double>(i) / static_cast<double>(n - 1) : 0.0;
            double env = 1.0;
            if (i < attack && !seg.legato) env = static_cast<double>(i) / attack;
            if (n - i <= release && !next_legato) env = std::min(env, static_cast<double>(n - i) / release);
            double y = 0.0, f0 = 0.0;
            if (seg.voiced) {
                const double t = static_cast<double>(i) / fs;
                const double midi = seg.midi_start + (seg.midi_end - seg.midi_start) * u +
                                    seg.vib_cents / 100.0 * std::sin(2 * kPi * seg.vib_hz * t);
                f0 = midi_to_hz(midi);
                phase += 2 * kPi * f0 / fs;
                if (phase > 2 * kPi * 1e6) phase = std::fmod(phase, 2 * kPi);
                const int harmonics = static_cast<int>(max_harm_hz / f0);
                for (int k = 1; k <= harmonics; ++k)
                    y += vowel_envelope(k * f0, model) / k * std::sin(k * phase);
                y = 0.25 * env * y + 0.002 * noise(rng);  // breath
            } else {
                y = 0.08 * env * sib.process(noise(rng));
            }
            s.audio.push_back(static_cast<float>(y));
            s.f0.push_back(f0);
            s.target.push_back(seg.voiced ? seg.target : -1);
            s.onset.push_back(i == 0 && seg.voiced && !seg.legato ? 1 : 0);
        }
    }
    return s;
}

std::vector<Scenario> builtin_scenarios() {
    const int minor = static_cast<int>(ScaleType::NaturalMinor);
    const int A = 9;
    std::vector<Scenario> out;

    // Steady notes detuned by known offsets (A minor: A2 B2 C3 D3 E3 = 45 47 48 50 52).
    {
        Scenario sc{"steady_detuned", "Low male range, steady notes detuned -35..+40 cents", A, minor, {}};
        const int notes[] = {45, 48, 50, 52, 47, 45};
        const double off[] = {-0.35, +0.25, -0.15, +0.40, -0.30, +0.20};
        for (int i = 0; i < 6; ++i) {
            sc.segments.push_back({0.45, notes[i] + off[i], notes[i] + off[i], 0, 5.5, true, notes[i]});
            sc.segments.push_back({0.06, 0, 0, 0, 0, false, -1});
        }
        out.push_back(sc);
    }
    // Same in a higher (tenor / female rap) register.
    {
        Scenario sc{"steady_high", "A3-E4 range, steady notes detuned", A, minor, {}};
        const int notes[] = {57, 60, 62, 64, 59, 57};
        const double off[] = {+0.30, -0.25, +0.35, -0.20, +0.15, -0.40};
        for (int i = 0; i < 6; ++i) {
            sc.segments.push_back({0.40, notes[i] + off[i], notes[i] + off[i], 0, 5.5, true, notes[i]});
            sc.segments.push_back({0.05, 0, 0, 0, 0, false, -1});
        }
        out.push_back(sc);
    }
    // Vibrato around detuned centres (hard-tune should flatten it).
    {
        Scenario sc{"vibrato", "Sustained notes, 5.5 Hz +/-30 cent vibrato, centres detuned", A, minor, {}};
        const int notes[] = {52, 55, 57, 53};
        const double off[] = {+0.10, -0.12, +0.08, -0.10};
        for (int i = 0; i < 4; ++i) {
            sc.segments.push_back({0.80, notes[i] + off[i], notes[i] + off[i], 30, 5.5, true, notes[i]});
            sc.segments.push_back({0.08, 0, 0, 0, 0, false, -1});
        }
        out.push_back(sc);
    }
    // Rap-like fast flow: 70-140 ms syllables, consonant gaps, falling contours.
    {
        Scenario sc{"fast_flow", "Rap cadence: short syllables with small falling glides and consonants", A, minor, {}};
        std::mt19937 rng(7);
        std::uniform_real_distribution<double> dur(0.07, 0.14), gap(0.025, 0.06), det(-0.35, 0.35);
        const int pool[] = {45, 47, 48, 50, 52};
        for (int i = 0; i < 28; ++i) {
            const int note = pool[rng() % 5];
            const double d = det(rng);
            // Small fall within the syllable, staying inside +/-45 cents of the note.
            const double start = note + std::clamp(d + 0.08, -0.45, 0.45);
            const double end = note + std::clamp(d - 0.08, -0.45, 0.45);
            sc.segments.push_back({dur(rng), start, end, 0, 5.5, true, note});
            sc.segments.push_back({gap(rng), 0, 0, 0, 0, false, -1});
        }
        out.push_back(sc);
    }
    // Scoops: each note starts ~90 c flat and glides up over 160 ms into a
    // held note 15 c flat (legato), the classic expressive onset that Flex
    // should leave alone. Notes have no in-scale neighbour a semitone below,
    // so the target is unambiguous throughout.
    {
        Scenario sc{"scoop", "Onset scoops from -90 c into held notes at -15 c", A, minor, {}};
        const int notes[] = {52, 47, 45, 52};
        for (int note : notes) {
            Segment glide{0.16, note - 0.90, note - 0.15, 0, 5.5, true, note};
            Segment hold{0.34, note - 0.15, note - 0.15, 0, 5.5, true, note};
            hold.legato = true;
            sc.segments.push_back(glide);
            sc.segments.push_back(hold);
            sc.segments.push_back({0.08, 0, 0, 0, 0, false, -1});
        }
        out.push_back(sc);
    }
    // Very low voice (bass rapper), close to the 70 Hz floor.
    {
        Scenario sc{"low_voice", "E2-A2 range (82-110 Hz), detuned", A, minor, {}};
        const int notes[] = {40, 43, 45, 41, 40};
        const double off[] = {+0.30, -0.30, +0.20, -0.25, +0.35};
        for (int i = 0; i < 5; ++i) {
            sc.segments.push_back({0.45, notes[i] + off[i], notes[i] + off[i], 0, 5.5, true, notes[i]});
            sc.segments.push_back({0.06, 0, 0, 0, 0, false, -1});
        }
        out.push_back(sc);
    }
    return out;
}

}  // namespace tune
