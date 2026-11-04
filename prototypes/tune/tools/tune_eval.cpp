// Tuner benchmark.
//
// Synthetic mode (default): runs the built-in scenarios and prints pitch
// accuracy, onset settle time, HF artifact energy, splice rate and CPU.
//
//   tune_eval [--out DIR] [--retune MS ...] [--fs HZ]
//
// Real-clip mode: scores WAV files in a local, git-ignored folder (see
// data/README.md). There is no ground truth, so it reports how close voiced
// frames sit to the scale before and after tuning.
//
//   tune_eval --clips DIR [--out DIR] [--retune MS ...] [--key A --scale minor]
//
// DIR/manifest.csv (optional) overrides the key per file:
//   file,key,scale[,low_hz,high_hz]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "tune/metrics.hpp"
#include "tune/scale.hpp"
#include "tune/synth.hpp"
#include "tune/tuner.hpp"
#include "tune/wav.hpp"

using namespace tune;
namespace fs = std::filesystem;

namespace {

struct ClipSettings {
    int tonic = 9;
    ScaleType scale = ScaleType::NaturalMinor;
    double low_hz = 70.0, high_hz = 800.0;
};

FormantMode g_formant_mode = FormantMode::Auto;

std::vector<float> run_tuner(const std::vector<float>& x, double rate, const ClipSettings& cs,
                             double retune, Tuner& tuner, double& ns_per_sample) {
    DetectorConfig det;
    det.min_hz = cs.low_hz;
    det.max_hz = cs.high_hz;
    tuner.prepare(rate, det);
    TunerParams p;
    p.tonic = cs.tonic;
    p.scale = cs.scale;
    p.retune_ms = retune;
    p.formant_mode = g_formant_mode;
    tuner.set_params(p);
    std::vector<float> y(x.size());
    const auto t0 = std::chrono::steady_clock::now();
    tuner.process(x.data(), y.data(), y.size());
    const auto t1 = std::chrono::steady_clock::now();
    ns_per_sample = y.empty() ? 0.0
                              : std::chrono::duration<double, std::nano>(t1 - t0).count() / y.size();
    return y;
}

// Formant test (FR-TUNE-6): steady_detuned notes transposed by +/-3 and
// +/-5 semitones, Preserve vs Follow. Reports the envelope error against the
// original vowel envelope and against the envelope shifted with the pitch,
// and where F1/F2 land.
void formant_report(double rate) {
    Scenario sc;
    for (const auto& s : builtin_scenarios())
        if (s.name == "steady_detuned") sc = s;
    const SynthSignal sig = synth_voice(sc.segments, rate, 1);
    const auto in_env = [](double f) { return vowel_envelope(f); };
    std::printf("\n| transpose st | mode | LSD vs original env dB | LSD vs pitch-shifted env dB | F1 Hz (orig 730) | F2 Hz (orig 1090) | ns/sample |\n");
    std::printf("|---|---|---|---|---|---|---|\n");
    for (double st : {-5.0, -3.0, 0.0, 3.0, 5.0}) {
        for (bool preserve : {true, false}) {
            Tuner tuner;
            tuner.prepare(rate);
            TunerParams p;
            p.tonic = sc.tonic;
            p.scale = static_cast<ScaleType>(sc.scale);
            p.retune_ms = 0.0;
            p.transpose_semitones = st;
            p.formant_mode = preserve ? FormantMode::Preserve : FormantMode::Follow;
            tuner.set_params(p);
            std::vector<float> y(sig.audio.size());
            const auto t0 = std::chrono::steady_clock::now();
            tuner.process(sig.audio.data(), y.data(), y.size());
            const auto t1 = std::chrono::steady_clock::now();
            const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / y.size();
            const double ratio = std::exp2(st / 12.0);
            const auto shifted_env = [ratio](double f) { return vowel_envelope(f / ratio); };
            double lsd_o = 0, lsd_s = 0, f1 = 0, f2 = 0;
            int n = 0;
            std::size_t i = 0;
            while (i < sig.target.size()) {
                if (sig.target[i] < 0 || !sig.onset[i]) { ++i; continue; }
                std::size_t j = i + 1;
                while (j < sig.target.size() && sig.target[j] == sig.target[i] && !sig.onset[j]) ++j;
                const std::size_t start = i + static_cast<std::size_t>(0.15 * rate) + tuner.latency_samples();
                const std::size_t len = static_cast<std::size_t>(0.2 * rate);
                if (start + len < j + tuner.latency_samples()) {
                    const double f_out = midi_to_hz(sig.target[i] + st);
                    const EnvelopeFit a = fit_envelope(y, rate, start, len, f_out, in_env);
                    const EnvelopeFit b = fit_envelope(y, rate, start, len, f_out, shifted_env);
                    lsd_o += a.lsd_db; lsd_s += b.lsd_db; f1 += a.f1_hz; f2 += a.f2_hz; ++n;
                }
                i = j;
            }
            if (n == 0) continue;
            std::printf("| %+.0f | %s | %.2f | %.2f | %.0f | %.0f | %.0f |\n", st,
                        preserve ? "preserve" : "follow", lsd_o / n, lsd_s / n, f1 / n, f2 / n, ns);
        }
    }
}

int synthetic(const std::string& out_dir, const std::vector<double>& retunes, double rate) {
    std::printf("fs = %.0f Hz, formant mode %s\n\n", rate,
                g_formant_mode == FormantMode::Auto ? "auto" : g_formant_mode == FormantMode::Preserve ? "preserve" : "follow");
    std::printf("| scenario | signal | retune ms | frames | median c | p95 c | within 10c | >50c | octave | settle med ms | settle p90 ms | unsettled | HF dB | jumps/s | ns/sample |\n");
    std::printf("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n");
    for (const auto& sc : builtin_scenarios()) {
        const SynthSignal sig = synth_voice(sc.segments, rate, 1);
        const double dur = sig.audio.size() / rate;
        const PitchStats in_stats = score_pitch(sig.audio, sig, 0);
        const double in_hf = hf_energy_db(sig.audio, sig, 0);
        std::printf("| %s | input | – | %s | – | – | – | %.1f | – | – |\n", sc.name.c_str(),
                    format_stats(in_stats).c_str(), in_hf);
        if (!out_dir.empty())
            write_wav(out_dir + "/" + sc.name + "_in.wav", Audio{static_cast<int>(rate), 1, sig.audio});
        ClipSettings cs;
        cs.tonic = sc.tonic;
        cs.scale = static_cast<ScaleType>(sc.scale);
        for (double rt : retunes) {
            Tuner tuner;
            double ns = 0.0;
            const auto out = run_tuner(sig.audio, rate, cs, rt, tuner, ns);
            const int lat = tuner.latency_samples();
            const PitchStats st = score_pitch(out, sig, lat);
            const SettleStats se = score_settle(out, sig, lat);
            const double hf = hf_energy_db(out, sig, lat);
            std::printf("| %s | tuned | %.0f | %s | %.1f | %.1f | %d/%d | %.1f | %.1f | %.0f |\n",
                        sc.name.c_str(), rt, format_stats(st).c_str(), se.median_ms, se.p90_ms,
                        se.unsettled, se.segments, hf, tuner.jump_count() / dur, ns);
            if (!out_dir.empty()) {
                char name[512];
                std::snprintf(name, sizeof name, "%s/%s_retune%.0f.wav", out_dir.c_str(),
                              sc.name.c_str(), rt);
                write_wav(name, Audio{static_cast<int>(rate), 1, out});
            }
        }
    }
    formant_report(rate);
    return 0;
}

std::map<std::string, ClipSettings> read_manifest(const fs::path& dir, const ClipSettings& def) {
    std::map<std::string, ClipSettings> m;
    std::ifstream f(dir / "manifest.csv");
    std::string line;
    int lineno = 0;
    while (std::getline(f, line)) {
        ++lineno;
        if (line.empty() || line[0] == '#') continue;
        std::vector<std::string> cols;
        std::stringstream ss(line);
        std::string c;
        while (std::getline(ss, c, ',')) cols.push_back(c);
        if (cols.size() < 3 || cols[0] == "file") continue;
        ClipSettings cs = def;
        const auto t = parse_tonic(cols[1]);
        const auto sc = parse_scale(cols[2]);
        if (!t || !sc) {
            std::fprintf(stderr, "manifest.csv:%d: bad key/scale, using defaults\n", lineno);
        } else {
            cs.tonic = *t;
            cs.scale = *sc;
        }
        if (cols.size() >= 5) {
            cs.low_hz = std::stod(cols[3]);
            cs.high_hz = std::stod(cols[4]);
        }
        m[cols[0]] = cs;
    }
    return m;
}

int clips(const std::string& dir, const std::string& out_dir, const std::vector<double>& retunes,
          const ClipSettings& def) {
    if (!fs::is_directory(dir)) {
        std::fprintf(stderr, "not a directory: %s\n", dir.c_str());
        return 2;
    }
    const auto manifest = read_manifest(dir, def);
    std::vector<fs::path> files;
    for (const auto& e : fs::directory_iterator(dir))
        if (e.is_regular_file() && e.path().extension() == ".wav") files.push_back(e.path());
    std::sort(files.begin(), files.end());
    if (files.empty()) {
        std::fprintf(stderr, "no .wav files in %s (see data/README.md)\n", dir.c_str());
        return 1;
    }
    std::printf("| clip | key | signal | retune ms | voiced frames | median c to scale | p95 c | within 10c | >50c | splices/s | ns/sample |\n");
    std::printf("|---|---|---|---|---|---|---|---|---|---|---|\n");
    for (const auto& path : files) {
        const std::string name = path.filename().string();
        const auto it = manifest.find(name);
        const ClipSettings cs = it != manifest.end() ? it->second : def;
        Audio a;
        try {
            a = read_wav(path.string());
        } catch (const std::exception& e) {
            std::fprintf(stderr, "skip %s: %s\n", name.c_str(), e.what());
            continue;
        }
        const auto x = to_mono(a);
        const unsigned mask = Scale::make(cs.tonic, cs.scale).mask;
        static const char* kNames[12] = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"};
        const std::string key = std::string(kNames[cs.tonic]) + (it != manifest.end() ? "" : " (default)");
        const PitchStats in = score_against_scale(x, a.sample_rate, mask, cs.low_hz, cs.high_hz);
        std::printf("| %s | %s | input | – | %d | %.1f | %.1f | %.1f%% | %.1f%% | – | – |\n",
                    name.c_str(), key.c_str(), in.frames, in.median_abs_cents, in.p95_abs_cents,
                    in.within10_pct, in.gross50_pct);
        for (double rt : retunes) {
            Tuner tuner;
            double ns = 0.0;
            const auto y = run_tuner(x, a.sample_rate, cs, rt, tuner, ns);
            const PitchStats st = score_against_scale(y, a.sample_rate, mask, cs.low_hz, cs.high_hz);
            const double dur = x.size() / static_cast<double>(a.sample_rate);
            std::printf("| %s | %s | tuned | %.0f | %d | %.1f | %.1f | %.1f%% | %.1f%% | %.1f | %.0f |\n",
                        name.c_str(), key.c_str(), rt, st.frames, st.median_abs_cents,
                        st.p95_abs_cents, st.within10_pct, st.gross50_pct,
                        tuner.jump_count() / dur, ns);
            if (!out_dir.empty()) {
                char out[1024];
                std::snprintf(out, sizeof out, "%s/%s_retune%.0f.wav", out_dir.c_str(),
                              path.stem().string().c_str(), rt);
                write_wav(out, Audio{a.sample_rate, 1, y});
            }
        }
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::string out_dir, clip_dir;
    std::vector<double> retunes;
    double rate = 48000.0;
    ClipSettings def;
    for (int i = 1; i < argc; ++i) {
        const bool has = i + 1 < argc;
        if (!std::strcmp(argv[i], "--out") && has) out_dir = argv[++i];
        else if (!std::strcmp(argv[i], "--retune") && has) retunes.push_back(std::atof(argv[++i]));
        else if (!std::strcmp(argv[i], "--fs") && has) rate = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--clips") && has) clip_dir = argv[++i];
        else if (!std::strcmp(argv[i], "--formant") && has) {
            const std::string m = argv[++i];
            if (m == "auto") g_formant_mode = FormantMode::Auto;
            else if (m == "preserve") g_formant_mode = FormantMode::Preserve;
            else if (m == "follow") g_formant_mode = FormantMode::Follow;
            else { std::fprintf(stderr, "bad --formant (auto|preserve|follow)\n"); return 2; }
        }
        else if (!std::strcmp(argv[i], "--key") && has) {
            const auto t = parse_tonic(argv[++i]);
            if (!t) { std::fprintf(stderr, "bad key\n"); return 2; }
            def.tonic = *t;
        } else if (!std::strcmp(argv[i], "--scale") && has) {
            const auto s = parse_scale(argv[++i]);
            if (!s) { std::fprintf(stderr, "bad scale\n"); return 2; }
            def.scale = *s;
        } else {
            std::fprintf(stderr,
                         "usage: %s [--out DIR] [--retune MS]... [--fs HZ] [--formant auto|preserve|follow]\n"
                         "       %s --clips DIR [--out DIR] [--retune MS]... [--key A] [--scale minor]\n",
                         argv[0], argv[0]);
            return 2;
        }
    }
    if (retunes.empty()) retunes = {0.0, 50.0};
    if (!out_dir.empty()) fs::create_directories(out_dir);
    try {
        return clip_dir.empty() ? synthetic(out_dir, retunes, rate)
                                : clips(clip_dir, out_dir, retunes, def);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
