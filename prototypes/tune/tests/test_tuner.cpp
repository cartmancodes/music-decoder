#include <infra/doctest.hpp>

#include <algorithm>
#include <initializer_list>

#include <cmath>
#include <vector>

#include "alloc_counter.hpp"
#include "tune/metrics.hpp"
#include "tune/synth.hpp"
#include "tune/tuner.hpp"

using namespace tune;

namespace {
SynthSignal scenario_signal(const std::string& name, Scenario* out = nullptr) {
    for (const auto& sc : builtin_scenarios()) {
        if (sc.name == name) {
            if (out) *out = sc;
            return synth_voice(sc.segments, 48000.0, 1);
        }
    }
    FAIL("unknown scenario " << name);
    return {};
}

std::vector<float> run(Tuner& t, const std::vector<float>& x, std::size_t block) {
    std::vector<float> y(x.size());
    for (std::size_t i = 0; i < x.size(); i += block) {
        const std::size_t n = std::min(block, x.size() - i);
        t.process(x.data() + i, y.data() + i, n);
    }
    return y;
}
}  // namespace

TEST_CASE("reported latency is 2 ms at every sample rate (NFR-1 budget)") {
    for (double fs : {44100.0, 48000.0, 96000.0, 192000.0}) {
        Tuner t;
        t.prepare(fs);
        const double ms = 1000.0 * t.latency_samples() / fs;
        CHECK(ms == doctest::Approx(2.0).epsilon(0.02));
        CHECK(ms <= 3.0);
    }
}

TEST_CASE("output does not depend on host block size (NFR-6)") {
    Scenario sc;
    const SynthSignal sig = scenario_signal("fast_flow", &sc);
    TunerParams p;
    p.tonic = sc.tonic;
    p.scale = static_cast<ScaleType>(sc.scale);
    std::vector<float> ref;
    for (std::size_t block : {std::size_t{1}, std::size_t{7}, std::size_t{64}, std::size_t{512}, std::size_t{4096}}) {
        Tuner t;
        t.prepare(48000.0);
        t.set_params(p);
        auto y = run(t, sig.audio, block);
        if (ref.empty()) ref = y;
        else REQUIRE(y == ref);
    }
}

TEST_CASE("tuner process() does not allocate (NFR-8 proxy)") {
    const SynthSignal sig = scenario_signal("steady_detuned");
    Tuner t;
    t.prepare(48000.0);
    TunerParams p;
    p.tonic = 9;
    p.scale = ScaleType::NaturalMinor;
    t.set_params(p);
    std::vector<float> y(sig.audio.size());
    tune_test::AllocScope scope;
    t.process(sig.audio.data(), y.data(), y.size());
    t.set_params(p);  // parameter changes are real-time safe too
    CHECK(scope.count() == 0);
}

TEST_CASE("hard-tune corrects steady detuned notes (FR-TUNE-2/4, spike thresholds)") {
    for (const char* name : {"steady_detuned", "steady_high"}) {
        Scenario sc;
        const SynthSignal sig = scenario_signal(name, &sc);
        Tuner t;
        t.prepare(48000.0);
        TunerParams p;
        p.tonic = sc.tonic;
        p.scale = static_cast<ScaleType>(sc.scale);
        p.retune_ms = 0.0;
        t.set_params(p);
        auto y = run(t, sig.audio, 256);
        const PitchStats in = score_pitch(sig.audio, sig, 0);
        const PitchStats out = score_pitch(y, sig, t.latency_samples());
        INFO(name << " in: " << format_stats(in) << "  out: " << format_stats(out));
        CHECK(in.median_abs_cents > 15.0);   // the input really is out of tune
        CHECK(out.median_abs_cents <= 5.0);  // spec §15.5 target
        CHECK(out.gross50_pct <= 2.0);       // FR-TUNE-2
        CHECK(out.octave_pct <= 0.5);        // FR-TUNE-1
    }
}

namespace {
PitchStats tuned_stats(const char* name, double retune_ms) {
    Scenario sc;
    const SynthSignal sig = scenario_signal(name, &sc);
    Tuner t;
    t.prepare(48000.0);
    TunerParams p;
    p.tonic = sc.tonic;
    p.scale = static_cast<ScaleType>(sc.scale);
    p.retune_ms = retune_ms;
    t.set_params(p);
    auto y = run(t, sig.audio, 128);
    return score_pitch(y, sig, t.latency_samples());
}
}  // namespace

TEST_CASE("hard-tune flattens vibrato and fast rap syllables (period tracker)") {
    for (const char* name : {"vibrato", "fast_flow", "low_voice"}) {
        const PitchStats out = tuned_stats(name, 0.0);
        INFO(name << ": " << format_stats(out));
        CHECK(out.median_abs_cents <= 5.0);
        CHECK(out.within10_pct >= 95.0);
        CHECK(out.gross50_pct <= 2.0);
    }
}

TEST_CASE("slow retune keeps vibrato (natural mode behaviour)") {
    // Against the flat target, a 50 ms retune should leave most of the
    // +/-30 cent vibrato in place rather than flattening it.
    const PitchStats slow = tuned_stats("vibrato", 50.0);
    INFO(format_stats(slow));
    CHECK(slow.median_abs_cents >= 10.0);
    CHECK(slow.gross50_pct <= 2.0);
}

TEST_CASE("onset settle time, hard-tune (T1.2 regression guard)") {
    // Pinned from spike v0.1 measurements (onset fix: fast voicing end, YIN
    // acquisition, guarded re-anchor) with margin. v0 values in comments.
    struct Case { const char* name; double max_median_ms, max_p90_ms; };
    for (const Case c : {Case{"steady_detuned", 8.0, 15.0},   // v0.1: 3 / 10   (v0: 17 / 24)
                         Case{"steady_high", 5.0, 12.0},      // v0.1: 0 / 7    (v0: 22 / 33)
                         Case{"vibrato", 5.0, 13.0},          // v0.1: 0 / 8.4  (v0: 20 / 24)
                         Case{"fast_flow", 5.0, 10.0},        // v0.1: 1.2 / 6.1 (v0: 9.7 / 18)
                         Case{"low_voice", 17.0, 20.0}}) {    // v0.1: 12 / 14  (v0: 18 / 24)
        Scenario sc;
        const SynthSignal sig = scenario_signal(c.name, &sc);
        Tuner t;
        t.prepare(48000.0);
        TunerParams p;
        p.tonic = sc.tonic;
        p.scale = static_cast<ScaleType>(sc.scale);
        p.retune_ms = 0.0;
        t.set_params(p);
        auto y = run(t, sig.audio, 128);
        const SettleStats s = score_settle(y, sig, t.latency_samples());
        INFO(c.name << ": median " << s.median_ms << " ms, p90 " << s.p90_ms << " ms, unsettled "
                    << s.unsettled << "/" << s.segments);
        CHECK(s.unsettled == 0);
        CHECK(s.median_ms <= c.max_median_ms);
        CHECK(s.p90_ms <= c.max_p90_ms);
    }
}

TEST_CASE("scale-distance metric: tuned output sits on the scale") {
    Scenario sc;
    const SynthSignal sig = scenario_signal("steady_detuned", &sc);
    const unsigned mask = Scale::make(sc.tonic, static_cast<ScaleType>(sc.scale)).mask;
    Tuner t;
    t.prepare(48000.0);
    TunerParams p;
    p.tonic = sc.tonic;
    p.scale = static_cast<ScaleType>(sc.scale);
    t.set_params(p);
    auto y = run(t, sig.audio, 256);
    const PitchStats in = score_against_scale(sig.audio, 48000.0, mask);
    const PitchStats out = score_against_scale(y, 48000.0, mask);
    INFO("in " << format_stats(in) << " out " << format_stats(out));
    CHECK(in.median_abs_cents > 15.0);
    CHECK(out.median_abs_cents < 3.0);
}

TEST_CASE("accuracy and settle hold at 44.1 and 96 kHz (window caps are durations)") {
    for (double fs : {44100.0, 96000.0}) {
        for (const char* name : {"low_voice", "fast_flow"}) {
            Scenario sc;
            for (const auto& s : builtin_scenarios())
                if (s.name == name) sc = s;
            const SynthSignal sig = synth_voice(sc.segments, fs, 1);
            Tuner t;
            t.prepare(fs);
            TunerParams p;
            p.tonic = sc.tonic;
            p.scale = static_cast<ScaleType>(sc.scale);
            t.set_params(p);
            auto y = run(t, sig.audio, 256);
            const PitchStats st = score_pitch(y, sig, t.latency_samples());
            const SettleStats se = score_settle(y, sig, t.latency_samples());
            INFO(name << " @ " << fs << ": " << format_stats(st) << " settle " << se.median_ms << "/"
                      << se.p90_ms);
            CHECK(st.median_abs_cents <= 1.0);
            CHECK(st.p95_abs_cents <= 2.0);
            CHECK(se.unsettled == 0);
            CHECK(se.p90_ms <= 20.0);
        }
    }
}
