#include <infra/doctest.hpp>

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <random>
#include <string>
#include <vector>

#include "alloc_counter.hpp"
#include "tune/formant.hpp"
#include "tune/lpc.hpp"
#include "tune/metrics.hpp"
#include "tune/synth.hpp"
#include "tune/tuner.hpp"

using namespace tune;

namespace {
Scenario scenario(const std::string& name) {
    for (const auto& s : builtin_scenarios())
        if (s.name == name) return s;
    FAIL("unknown scenario " << name);
    return {};
}

// Mean F1 (and envelope error) over the steady part of each note.
struct Fit { double f1 = 0, lsd = 0; };
Fit fit(const SynthSignal& sig, const std::vector<float>& y, int lat, double st, VoiceModel m) {
    Fit out;
    int n = 0;
    std::size_t i = 0;
    while (i < sig.target.size()) {
        if (sig.target[i] < 0 || !sig.onset[i]) { ++i; continue; }
        std::size_t j = i + 1;
        while (j < sig.target.size() && sig.target[j] == sig.target[i] && !sig.onset[j]) ++j;
        const std::size_t start = i + static_cast<std::size_t>(0.15 * sig.sample_rate) + lat;
        const std::size_t len = static_cast<std::size_t>(0.2 * sig.sample_rate);
        if (start + len < j) {
            const auto e = fit_envelope(y, sig.sample_rate, start, len, midi_to_hz(sig.target[i] + st),
                                        [m](double f) { return vowel_envelope(f, m); });
            out.f1 += e.f1_hz;
            out.lsd += e.lsd_db;
            ++n;
        }
        i = j;
    }
    if (n) { out.f1 /= n; out.lsd /= n; }
    return out;
}

std::vector<float> run(const SynthSignal& sig, const Scenario& sc, double transpose, bool preserve,
                       int* lat = nullptr, double amount = 1.0) {
    Tuner t;
    t.prepare(sig.sample_rate);
    TunerParams p;
    p.tonic = sc.tonic;
    p.scale = static_cast<ScaleType>(sc.scale);
    p.transpose_semitones = transpose;
    p.formant_mode = preserve ? FormantMode::Preserve : FormantMode::Follow;
    p.amount = amount;
    t.set_params(p);
    std::vector<float> y(sig.audio.size());
    t.process(sig.audio.data(), y.data(), y.size());
    if (lat) *lat = t.latency_samples();
    return y;
}
}  // namespace

TEST_CASE("Levinson recovers a known AR(2) model") {
    // x[n] = 1.3 x[n-1] - 0.6 x[n-2] + w[n]  ->  A(z) = 1 - 1.3 z^-1 + 0.6 z^-2
    std::mt19937 rng(3);
    std::normal_distribution<double> w(0.0, 1.0);
    std::vector<double> x(200000);
    for (std::size_t n = 2; n < x.size(); ++n) x[n] = 1.3 * x[n - 1] - 0.6 * x[n - 2] + w(rng);
    double r[3], k[2], a[3], tmp[3];
    autocorrelate(x.data(), static_cast<int>(x.size()), 2, r);
    levinson(r, 2, k, a, tmp);
    CHECK(a[1] == doctest::Approx(-1.3).epsilon(0.01));
    CHECK(a[2] == doctest::Approx(0.6).epsilon(0.02));
}

TEST_CASE("whiten -> recolour is an exact identity with time-varying coefficients") {
    const SynthSignal sig = synth_voice(scenario("fast_flow").segments, 48000.0, 1);
    for (double lambda : {0.0, 0.5, -1.0 /* Bark default */}) {
        FormantConfig cfg;
        cfg.lambda = lambda;
        FormantChain fc;
        fc.prepare(48000.0, cfg);
        double worst = 0.0;
        for (std::size_t n = 0; n < sig.audio.size(); ++n) {
            const float e = fc.whiten(sig.audio[n]);
            const float y = fc.recolour(e, static_cast<double>(n));
            worst = std::max(worst, static_cast<double>(std::abs(y - sig.audio[n])));
        }
        INFO("lambda " << lambda << " worst error " << worst);
        CHECK(worst < 1e-4);
    }
}

TEST_CASE("with correction off, the tuner with formant chain is an exact delay (FR-LAT-1)") {
    const Scenario sc = scenario("steady_detuned");
    const SynthSignal sig = synth_voice(sc.segments, 48000.0, 1);
    int lat = 0;
    const auto y = run(sig, sc, 0.0, true, &lat, /*amount=*/0.0);
    double worst = 0.0;
    for (std::size_t n = lat; n < y.size(); ++n)
        worst = std::max(worst, static_cast<double>(std::abs(y[n] - sig.audio[n - lat])));
    INFO("worst error " << worst);
    CHECK(worst < 1e-4);
}

TEST_CASE("formants stay put under transpose (FR-TUNE-6) on both synthetic voice models") {
    const Scenario sc = scenario("steady_detuned");
    for (VoiceModel m : {VoiceModel::Resonator, VoiceModel::Gaussian}) {
        const SynthSignal sig = synth_voice(sc.segments, 48000.0, 1, m);
        int lat = 0;
        const Fit base = fit(sig, run(sig, sc, 0.0, true, &lat), lat, 0.0, m);
        for (double st : {-3.0, 3.0, 5.0}) {
            const Fit keep = fit(sig, run(sig, sc, st, true, &lat), lat, st, m);
            const Fit follow = fit(sig, run(sig, sc, st, false, &lat), lat, st, m);
            const double moved_keep = std::abs(keep.f1 / base.f1 - 1.0);
            const double moved_follow = std::abs(follow.f1 / base.f1 - 1.0);
            INFO((m == VoiceModel::Resonator ? "resonator" : "gaussian")
                 << " " << st << " st: F1 base " << base.f1 << " preserve " << keep.f1 << " follow "
                 << follow.f1 << " | LSD preserve " << keep.lsd << " follow " << follow.lsd);
            CHECK(moved_keep < 0.05);          // FR-TUNE-6
            CHECK(moved_follow > 0.08);        // sanity: the test can see formant movement
            // Shifting down samples the envelope between the original
            // harmonics, where sharp resonance peaks are unobservable, so only
            // "better than follow" is required; F1 position is the spec criterion.
            CHECK(keep.lsd < follow.lsd);
        }
    }
}

TEST_CASE("formant chain does not allocate after prepare") {
    const SynthSignal sig = synth_voice(scenario("fast_flow").segments, 48000.0, 1);
    FormantChain fc;
    fc.prepare(48000.0);
    tune_test::AllocScope scope;
    double acc = 0.0;
    for (std::size_t n = 0; n < sig.audio.size(); ++n)
        acc += fc.recolour(fc.whiten(sig.audio[n]), static_cast<double>(n) - 96.0);
    CHECK(scope.count() == 0);
    CHECK(std::isfinite(acc));
}

TEST_CASE("Auto formant mode preserves only when transposing") {
    Tuner t;
    t.prepare(48000.0);
    TunerParams p;
    t.set_params(p);
    CHECK_FALSE(t.preserving());
    p.transpose_semitones = 0.4;
    t.set_params(p);
    CHECK_FALSE(t.preserving());
    p.transpose_semitones = -2.0;
    t.set_params(p);
    CHECK(t.preserving());
    p.formant_mode = FormantMode::Follow;
    t.set_params(p);
    CHECK_FALSE(t.preserving());
    p.transpose_semitones = 0.0;
    p.formant_mode = FormantMode::Preserve;
    t.set_params(p);
    CHECK(t.preserving());
}

TEST_CASE("Preserve mode pitch cost stays bounded (spike v0.2 pins)") {
    // Measured: vibrato hard-tune 2.3 / 5.3 c (follow 0.7 / 3.1), settle
    // 8 / 13.4 ms (follow 0 / 8.4). Pins with margin; see spike report.
    const Scenario sc = scenario("vibrato");
    const SynthSignal sig = synth_voice(sc.segments, 48000.0, 1);
    int lat = 0;
    const auto y = run(sig, sc, 0.0, true, &lat);
    const PitchStats st = score_pitch(y, sig, lat);
    const SettleStats se = score_settle(y, sig, lat);
    INFO(format_stats(st) << " settle " << se.median_ms << "/" << se.p90_ms);
    CHECK(st.median_abs_cents <= 4.0);
    CHECK(st.p95_abs_cents <= 8.0);
    CHECK(se.unsettled == 0);
    CHECK(se.median_ms <= 12.0);
    CHECK(se.p90_ms <= 18.0);
}
