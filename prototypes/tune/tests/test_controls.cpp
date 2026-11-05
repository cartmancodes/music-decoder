// Spec §6.3 controls: vibrato gain, humanize, flex (spike v0.3).
#include <infra/doctest.hpp>

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <string>
#include <vector>

#include "alloc_counter.hpp"
#include "tune/metrics.hpp"
#include "tune/synth.hpp"
#include "tune/tuner.hpp"
#include "tune/yin_offline.hpp"

using namespace tune;

namespace {
Scenario scenario(const std::string& name) {
    for (const auto& s : builtin_scenarios())
        if (s.name == name) return s;
    FAIL("unknown scenario " << name);
    return {};
}

std::vector<float> run(const SynthSignal& sig, const Scenario& sc, TunerParams p, int* lat,
                       std::size_t block = 256) {
    Tuner t;
    t.prepare(sig.sample_rate);
    p.tonic = sc.tonic;
    p.scale = static_cast<ScaleType>(sc.scale);
    t.set_params(p);
    std::vector<float> y(sig.audio.size());
    for (std::size_t i = 0; i < y.size(); i += block)
        t.process(sig.audio.data() + i, y.data() + i, std::min(block, y.size() - i));
    if (lat) *lat = t.latency_samples();
    return y;
}

// Scoop metrics: mean |output - input| (cents) 25-55 ms after each onset
// (input still > ~65 c flat), and mean |output - target| 200-450 ms after.
void scoop_metrics(const std::vector<float>& y, const SynthSignal& sig, int lat, double& early,
                   double& hold) {
    const auto fr = yin_track(y, sig.sample_rate, 60.0, 1000.0, 0.002);
    double e = 0, h = 0;
    int ne = 0, nh = 0;
    for (std::size_t i = 0; i < sig.target.size(); ++i) {
        if (sig.target[i] < 0 || !sig.onset[i]) continue;
        const double ton = i / sig.sample_rate;
        const int tg = sig.target[i];
        for (const auto& f : fr) {
            if (f.hz <= 0.0) continue;
            const double tin = f.t - lat / sig.sample_rate;
            const long si = std::lround(tin * sig.sample_rate);
            if (si < 0 || si >= static_cast<long>(sig.f0.size())) continue;
            if (tin >= ton + 0.025 && tin <= ton + 0.055 && sig.f0[si] > 0.0) {
                e += std::abs(1200.0 * std::log2(f.hz / sig.f0[si]));
                ++ne;
            }
            if (tin >= ton + 0.20 && tin <= ton + 0.45) {
                h += std::abs(1200.0 * std::log2(f.hz / midi_to_hz(tg)));
                ++nh;
            }
        }
    }
    early = ne ? e / ne : 0.0;
    hold = nh ? h / nh : 0.0;
}
}  // namespace

TEST_CASE("vibrato gain: 0 flattens, 1 keeps the singer's vibrato, 2 doubles it; centre stays corrected") {
    const Scenario sc = scenario("vibrato");
    const SynthSignal sig = synth_voice(sc.segments, 48000.0, 1);
    const VibratoStats in = score_vibrato(sig.audio, sig, 0);
    REQUIRE(in.depth_cents > 15.0);
    REQUIRE(in.centre_error_cents > 5.0);  // input centres really are detuned
    struct Case { double gain, min_ratio, max_ratio, max_centre; };
    for (const Case c : {Case{0.0, 0.0, 0.15, 2.0}, Case{1.0, 0.85, 1.35, 5.0},
                         Case{2.0, 1.8, 2.6, 8.0}}) {
        TunerParams p;
        p.vibrato = c.gain;
        int lat = 0;
        const auto y = run(sig, sc, p, &lat);
        const VibratoStats v = score_vibrato(y, sig, lat);
        const double ratio = v.depth_cents / in.depth_cents;
        INFO("gain " << c.gain << ": depth ratio " << ratio << ", centre error " << v.centre_error_cents);
        CHECK(ratio >= c.min_ratio);
        CHECK(ratio <= c.max_ratio);
        CHECK(v.centre_error_cents <= c.max_centre);
    }
}

TEST_CASE("humanize keeps sustained notes natural but short syllables tight") {
    {
        const Scenario sc = scenario("vibrato");
        const SynthSignal sig = synth_voice(sc.segments, 48000.0, 1);
        const VibratoStats in = score_vibrato(sig.audio, sig, 0);
        TunerParams p;
        p.humanize = 1.0;
        int lat = 0;
        const auto y = run(sig, sc, p, &lat);
        const VibratoStats v = score_vibrato(y, sig, lat);
        INFO("held notes, humanize 1: depth " << v.depth_cents << " (input " << in.depth_cents
                                               << "), centre error " << v.centre_error_cents);
        CHECK(v.depth_cents >= 0.7 * in.depth_cents);  // natural movement kept
        CHECK(v.centre_error_cents <= 3.0);            // still on pitch
    }
    {
        const Scenario sc = scenario("fast_flow");
        const SynthSignal sig = synth_voice(sc.segments, 48000.0, 1);
        TunerParams p;
        p.humanize = 1.0;
        int lat = 0;
        const auto y = run(sig, sc, p, &lat);
        const PitchStats st = score_pitch(y, sig, lat);
        const SettleStats se = score_settle(y, sig, lat);
        INFO("short syllables, humanize 1: " << format_stats(st) << " settle " << se.median_ms << "/"
                                              << se.p90_ms);
        CHECK(st.median_abs_cents <= 3.0);
        CHECK(st.within10_pct >= 95.0);
        CHECK(se.p90_ms <= 10.0);
    }
}

TEST_CASE("flex lets scoops through and still corrects the held note") {
    const Scenario sc = scenario("scoop");
    const SynthSignal sig = synth_voice(sc.segments, 48000.0, 1);
    double early = 0, hold = 0;
    int lat = 0;
    {
        TunerParams p;  // flex off: the scoop is pulled to the target
        scoop_metrics(run(sig, sc, p, &lat), sig, lat, early, hold);
        INFO("flex 0: early |out-in| " << early << ", hold |out-target| " << hold);
        CHECK(early >= 50.0);
        CHECK(hold <= 2.0);
    }
    {
        TunerParams p;
        p.flex = 0.3;
        scoop_metrics(run(sig, sc, p, &lat), sig, lat, early, hold);
        INFO("flex 0.3: early |out-in| " << early << ", hold |out-target| " << hold);
        CHECK(early <= 6.0);  // scoop passes through
        CHECK(hold <= 3.0);   // held note corrected
    }
}

TEST_CASE("controls: no allocation, block-size invariant, safe to change mid-stream") {
    const Scenario sc = scenario("vibrato");
    const SynthSignal sig = synth_voice(sc.segments, 48000.0, 1);
    TunerParams p;
    p.tonic = sc.tonic;
    p.scale = static_cast<ScaleType>(sc.scale);
    p.humanize = 0.6;
    p.flex = 0.4;
    p.vibrato = 1.3;
    p.retune_ms = 15.0;
    std::vector<float> ref;
    for (std::size_t block : {std::size_t{1}, std::size_t{64}, std::size_t{4096}}) {
        Tuner t;
        t.prepare(48000.0);
        t.set_params(p);
        std::vector<float> y(sig.audio.size());
        tune_test::AllocScope scope;
        for (std::size_t i = 0; i < y.size(); i += block)
            t.process(sig.audio.data() + i, y.data() + i, std::min(block, y.size() - i));
        CHECK(scope.count() == 0);
        if (ref.empty()) ref = y;
        else REQUIRE(y == ref);
    }
    // Automate every control per block: no allocation, finite output.
    Tuner t;
    t.prepare(48000.0);
    std::vector<float> y(sig.audio.size());
    tune_test::AllocScope scope;
    double acc = 0.0;
    for (std::size_t i = 0, b = 0; i < y.size(); i += 128, ++b) {
        TunerParams q = p;
        q.humanize = 0.5 + 0.5 * std::sin(0.01 * b);
        q.flex = 0.5 + 0.5 * std::sin(0.013 * b);
        q.vibrato = 1.0 + std::sin(0.017 * b);
        q.retune_ms = 20.0 + 20.0 * std::sin(0.019 * b);
        t.set_params(q);
        const std::size_t n = std::min<std::size_t>(128, y.size() - i);
        t.process(sig.audio.data() + i, y.data() + i, n);
        for (std::size_t k = 0; k < n; ++k) acc += y[i + k];
    }
    CHECK(scope.count() == 0);
    CHECK(std::isfinite(acc));
}
