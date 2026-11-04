#include <infra/doctest.hpp>

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <string>

#include "alloc_counter.hpp"
#include "tune/detector.hpp"
#include "tune/synth.hpp"

using namespace tune;

namespace {
SynthSignal signal_for(const std::string& name) {
    for (const auto& sc : builtin_scenarios())
        if (sc.name == name) return synth_voice(sc.segments, 48000.0, 1);
    FAIL("unknown scenario " << name);
    return {};
}
}  // namespace

TEST_CASE("each note is acquired fresh: first voiced pitch is the note's, not the previous one's") {
    for (const char* name : {"steady_detuned", "steady_high", "fast_flow", "low_voice"}) {
        const SynthSignal sig = signal_for(name);
        Detector d;
        d.prepare(48000.0, DetectorConfig{});
        int onsets = 0, fresh = 0, stale_at_onset = 0;
        bool waiting = false;
        for (std::size_t i = 0; i < sig.audio.size(); ++i) {
            if (sig.onset[i]) {
                ++onsets;
                waiting = true;
                if (d.voiced()) ++stale_at_onset;  // still voiced from the previous note
            }
            const double f = d.process(sig.audio[i]);
            if (waiting && f > 0.0 && sig.f0[i] > 0.0) {
                waiting = false;
                if (std::abs(1200.0 * std::log2(f / sig.f0[i])) <= 50.0) ++fresh;
            }
        }
        INFO(name << ": fresh " << fresh << "/" << onsets << ", voiced at onset " << stale_at_onset);
        CHECK(fresh == onsets);
        CHECK(stale_at_onset == 0);
    }
}

TEST_CASE("voicing drops inside consonant gaps") {
    const SynthSignal sig = signal_for("fast_flow");
    Detector d;
    d.prepare(48000.0, DetectorConfig{});
    std::size_t gap = 0, gap_voiced = 0;
    for (std::size_t i = 0; i < sig.audio.size(); ++i) {
        const double f = d.process(sig.audio[i]);
        if (sig.f0[i] == 0.0) {
            ++gap;
            if (f > 0.0) ++gap_voiced;
        }
    }
    const double pct = 100.0 * gap_voiced / gap;
    INFO("voiced during gaps: " << pct << "%");
    CHECK(pct <= 25.0);  // the tail of each note may hold into the gap briefly
}

TEST_CASE("detector does not allocate after prepare, incl. acquisition") {
    const SynthSignal sig = signal_for("fast_flow");
    Detector d;
    d.prepare(48000.0, DetectorConfig{});
    tune_test::AllocScope scope;
    double acc = 0.0;
    for (float x : sig.audio) acc += d.process(x);
    CHECK(scope.count() == 0);
    CHECK(std::isfinite(acc));
}
