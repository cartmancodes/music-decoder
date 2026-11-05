// Throat (spec §6.5): formant shift via a different re-colour warping.
#include <infra/doctest.hpp>

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <string>
#include <vector>

#include "alloc_counter.hpp"
#include "tune/formant.hpp"
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

std::vector<float> run(const SynthSignal& sig, const Scenario& sc, TunerParams p, int* lat,
                       bool* chain = nullptr) {
    Tuner t;
    t.prepare(sig.sample_rate);
    p.tonic = sc.tonic;
    p.scale = static_cast<ScaleType>(sc.scale);
    t.set_params(p);
    std::vector<float> y(sig.audio.size());
    t.process(sig.audio.data(), y.data(), y.size());
    if (lat) *lat = t.latency_samples();
    if (chain) *chain = t.chain_active();
    return y;
}

struct ThroatFit { double f1 = 0, lsd_shifted = 0, lsd_original = 0; };
ThroatFit fit(const SynthSignal& sig, const std::vector<float>& y, int lat, double scale, VoiceModel m) {
    ThroatFit out;
    int n = 0;
    std::size_t i = 0;
    while (i < sig.target.size()) {
        if (sig.target[i] < 0 || !sig.onset[i]) { ++i; continue; }
        std::size_t j = i + 1;
        while (j < sig.target.size() && sig.target[j] == sig.target[i] && !sig.onset[j]) ++j;
        const std::size_t start = i + static_cast<std::size_t>(0.15 * sig.sample_rate) + lat;
        const std::size_t len = static_cast<std::size_t>(0.2 * sig.sample_rate);
        if (start + len < j) {
            const double f0 = midi_to_hz(sig.target[i]);
            // Envelope comparison over the F1-F2 region (<= 2 kHz): above
            // ~2.5 kHz the model's noise floor (3e-2) dominates after a shift,
            // the same trade-off as Preserve (spike report v0.4).
            const auto a = fit_envelope(y, sig.sample_rate, start, len, f0,
                                        [m, scale](double f) { return vowel_envelope(f / scale, m); },
                                        2000.0, scale);
            const auto b = fit_envelope(y, sig.sample_rate, start, len, f0,
                                        [m](double f) { return vowel_envelope(f, m); }, 2000.0);
            out.f1 += a.f1_hz;
            out.lsd_shifted += a.lsd_db;
            out.lsd_original += b.lsd_db;
            ++n;
        }
        i = j;
    }
    if (n) { out.f1 /= n; out.lsd_shifted /= n; out.lsd_original /= n; }
    return out;
}
}  // namespace

TEST_CASE("re-colour scale 1 keeps the chain an exact identity; other scales stay stable") {
    const SynthSignal sig = synth_voice(scenario("fast_flow").segments, 48000.0, 1);
    for (double scale : {1.0, 0.84, 1.19, 1.41}) {
        FormantChain fc;
        fc.prepare(48000.0);
        fc.set_recolour_scale(scale);
        double worst = 0.0, peak = 0.0;
        for (std::size_t n = 0; n < sig.audio.size(); ++n) {
            const float y = fc.recolour(fc.whiten(sig.audio[n]), static_cast<double>(n));
            worst = std::max(worst, static_cast<double>(std::abs(y - sig.audio[n])));
            peak = std::max(peak, static_cast<double>(std::abs(y)));
        }
        INFO("scale " << scale << ": lambda_r " << fc.recolour_lambda() << ", max diff " << worst
                      << ", peak " << peak);
        REQUIRE(std::isfinite(peak));
        if (scale == 1.0) CHECK(worst < 1e-4);
        else CHECK(worst > 1e-3);  // the envelope really moved
        CHECK(peak < 4.0);         // stable, no blow-up
    }
}

TEST_CASE("throat moves the formants by the requested amount without touching pitch") {
    const Scenario sc = scenario("steady_detuned");
    for (VoiceModel m : {VoiceModel::Resonator, VoiceModel::Gaussian}) {
        const SynthSignal sig = synth_voice(sc.segments, 48000.0, 1, m);
        int lat = 0;
        TunerParams base_p;
        base_p.formant_mode = FormantMode::Preserve;
        const ThroatFit base = fit(sig, run(sig, sc, base_p, &lat), lat, 1.0, m);
        for (FormantMode mode : {FormantMode::Preserve, FormantMode::Auto}) {
            for (double st : {-3.0, 3.0}) {
                TunerParams p;
                p.formant_mode = mode;
                p.throat_semitones = st;
                bool chain = false;
                const auto y = run(sig, sc, p, &lat, &chain);
                const double scale = std::exp2(st / 12.0);
                const ThroatFit f = fit(sig, y, lat, scale, m);
                const PitchStats ps = score_pitch(y, sig, lat);
                INFO((m == VoiceModel::Resonator ? "resonator" : "gaussian")
                     << (mode == FormantMode::Preserve ? " preserve " : " auto ") << st << " st: F1 "
                     << f.f1 << " want " << base.f1 * scale << " | LSD shifted " << f.lsd_shifted
                     << " original " << f.lsd_original << " | pitch median " << ps.median_abs_cents);
                CHECK(chain);
                CHECK(std::abs(f.f1 / (base.f1 * scale) - 1.0) <= 0.08);
                CHECK(f.lsd_shifted < f.lsd_original);  // the envelope moved, not just F1
                CHECK(ps.median_abs_cents <= 1.0);       // pitch unaffected
            }
        }
    }
}

TEST_CASE("throat 0 changes nothing (chain stays off in Auto at transpose 0)") {
    const Scenario sc = scenario("vibrato");
    const SynthSignal sig = synth_voice(sc.segments, 48000.0, 1);
    TunerParams p;
    int lat = 0;
    bool chain = true;
    const auto a = run(sig, sc, p, &lat, &chain);
    CHECK_FALSE(chain);
    p.throat_semitones = 0.0;
    const auto b = run(sig, sc, p, &lat);
    CHECK(a == b);
}

TEST_CASE("sweeping throat live: no allocation, no zipper noise") {
    const Scenario sc = scenario("steady_detuned");
    const SynthSignal sig = synth_voice(sc.segments, 48000.0, 1);
    // Reference: a fixed throat of +1.5 st.
    TunerParams p;
    p.tonic = sc.tonic;
    p.scale = static_cast<ScaleType>(sc.scale);
    p.formant_mode = FormantMode::Preserve;
    p.throat_semitones = 1.5;
    int lat = 0;
    const auto fixed = run(sig, sc, p, &lat);
    // Sweep -3..+3 st at 2 Hz, updated every 64 samples.
    Tuner t;
    t.prepare(48000.0);
    t.set_params(p);
    std::vector<float> y(sig.audio.size());
    tune_test::AllocScope scope;
    for (std::size_t i = 0; i < y.size(); i += 64) {
        TunerParams q = p;
        q.throat_semitones = 3.0 * std::sin(2.0 * 3.14159265358979323846 * 2.0 * i / 48000.0);
        t.set_params(q);
        t.process(sig.audio.data() + i, y.data() + i, std::min<std::size_t>(64, y.size() - i));
    }
    CHECK(scope.count() == 0);
    const double hf_fixed = hf_energy_db(fixed, sig, lat);
    const double hf_sweep = hf_energy_db(y, sig, lat);
    INFO("HF energy fixed " << hf_fixed << " dB, sweep " << hf_sweep << " dB");
    CHECK(std::isfinite(hf_sweep));
    CHECK(hf_sweep <= hf_fixed + 3.0);
}
