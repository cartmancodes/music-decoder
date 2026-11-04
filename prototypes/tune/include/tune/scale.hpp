// Scales, key parsing and the hysteresis note decider (spec §6.2).
// Real-time safe: no allocation, no locks.
#pragma once

#include <cmath>
#include <cstdint>
#include <optional>
#include <string_view>

namespace tune {

enum class ScaleType {
    Major,
    NaturalMinor,
    HarmonicMinor,
    MelodicMinor,
    MajorPentatonic,
    MinorPentatonic,
    Blues,
    Chromatic,
};

// Bit i set = pitch class i (C = 0) is allowed.
struct Scale {
    std::uint16_t mask = 0x0FFF;

    static constexpr std::uint16_t intervals(ScaleType t) {
        switch (t) {
            case ScaleType::Major: return 0b1010'1011'0101;          // 0 2 4 5 7 9 11
            case ScaleType::NaturalMinor: return 0b0101'1010'1101;   // 0 2 3 5 7 8 10
            case ScaleType::HarmonicMinor: return 0b1001'1010'1101;  // 0 2 3 5 7 8 11
            case ScaleType::MelodicMinor: return 0b1010'1010'1101;   // 0 2 3 5 7 9 11
            case ScaleType::MajorPentatonic: return 0b0010'1001'0101;  // 0 2 4 7 9
            case ScaleType::MinorPentatonic: return 0b0100'1010'1001;  // 0 3 5 7 10
            case ScaleType::Blues: return 0b0100'1110'1001;          // 0 3 5 6 7 10
            case ScaleType::Chromatic: return 0x0FFF;
        }
        return 0x0FFF;
    }

    static constexpr Scale make(int tonic_pc, ScaleType t) {
        const std::uint16_t iv = intervals(t);
        std::uint16_t m = 0;
        for (int i = 0; i < 12; ++i)
            if (iv & (1u << i)) m |= static_cast<std::uint16_t>(1u << ((tonic_pc + i) % 12));
        return Scale{m};
    }

    constexpr bool allows(int pc) const { return (mask >> (((pc % 12) + 12) % 12)) & 1u; }
};

// "C", "C#", "Db", ... -> pitch class, or nullopt.
std::optional<int> parse_tonic(std::string_view s);
// "major", "minor"/"natural_minor", "harmonic_minor", "melodic_minor",
// "major_pentatonic", "minor_pentatonic", "blues", "chromatic".
std::optional<ScaleType> parse_scale(std::string_view s);

inline std::optional<int> parse_tonic(std::string_view s) {
    if (s.empty()) return std::nullopt;
    static constexpr int base[7] = {9, 11, 0, 2, 4, 5, 7};  // A B C D E F G
    char c = s[0];
    if (c >= 'a' && c <= 'g') c = static_cast<char>(c - 'a' + 'A');
    if (c < 'A' || c > 'G') return std::nullopt;
    int pc = base[c - 'A'];
    if (s.size() == 2) {
        if (s[1] == '#') pc += 1;
        else if (s[1] == 'b') pc -= 1;
        else return std::nullopt;
    } else if (s.size() > 2) {
        return std::nullopt;
    }
    return (pc + 12) % 12;
}

inline std::optional<ScaleType> parse_scale(std::string_view s) {
    if (s == "major") return ScaleType::Major;
    if (s == "minor" || s == "natural_minor") return ScaleType::NaturalMinor;
    if (s == "harmonic_minor") return ScaleType::HarmonicMinor;
    if (s == "melodic_minor") return ScaleType::MelodicMinor;
    if (s == "major_pentatonic") return ScaleType::MajorPentatonic;
    if (s == "minor_pentatonic") return ScaleType::MinorPentatonic;
    if (s == "blues") return ScaleType::Blues;
    if (s == "chromatic") return ScaleType::Chromatic;
    return std::nullopt;
}

// Nearest allowed MIDI note to `midi` (fractional input, integer result).
inline int nearest_allowed(const Scale& scale, double midi) {
    const int centre = static_cast<int>(std::lround(midi));
    int best = centre;
    double best_d = 1e9;
    for (int k = -6; k <= 6; ++k) {
        const int n = centre + k;
        if (!scale.allows(n)) continue;
        const double d = std::abs(midi - n);
        if (d < best_d) {
            best_d = d;
            best = n;
        }
    }
    return best;
}

// Quantizer with hysteresis: keeps the current target until another allowed
// note is closer to the input by more than `hysteresis_cents`.
class NoteDecider {
public:
    void set_scale(Scale s) {
        scale_ = s;
        if (current_ >= 0 && !scale_.allows(current_)) current_ = -1;
    }
    void set_hysteresis_cents(double h) { hysteresis_ = h / 100.0; }
    void reset() { current_ = -1; }

    // Returns the target MIDI note for input pitch `midi`.
    int decide(double midi) {
        const int candidate = nearest_allowed(scale_, midi);
        if (current_ < 0) {
            current_ = candidate;
        } else if (candidate != current_) {
            const double d_cur = std::abs(midi - current_);
            const double d_new = std::abs(midi - candidate);
            if (d_new + hysteresis_ < d_cur) current_ = candidate;
        }
        return current_;
    }

    int current() const { return current_; }

private:
    Scale scale_{};
    double hysteresis_ = 0.15;
    int current_ = -1;
};

inline double hz_to_midi(double hz) { return 69.0 + 12.0 * std::log2(hz / 440.0); }
inline double midi_to_hz(double m) { return 440.0 * std::exp2((m - 69.0) / 12.0); }

}  // namespace tune
