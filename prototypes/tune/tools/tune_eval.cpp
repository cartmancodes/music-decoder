// Runs the built-in synthetic scenarios through the tuner and prints
// objective metrics (markdown table). Optionally writes input/output WAVs.
//
//   tune_eval [--out DIR] [--retune MS ...] [--fs HZ]
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "tune/metrics.hpp"
#include "tune/synth.hpp"
#include "tune/tuner.hpp"
#include "tune/wav.hpp"

using namespace tune;

int main(int argc, char** argv) {
    std::string out_dir;
    std::vector<double> retunes;
    double fs = 48000.0;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--out") && i + 1 < argc) out_dir = argv[++i];
        else if (!std::strcmp(argv[i], "--retune") && i + 1 < argc) retunes.push_back(std::atof(argv[++i]));
        else if (!std::strcmp(argv[i], "--fs") && i + 1 < argc) fs = std::atof(argv[++i]);
        else {
            std::fprintf(stderr, "usage: %s [--out DIR] [--retune MS]... [--fs HZ]\n", argv[0]);
            return 2;
        }
    }
    if (retunes.empty()) retunes = {0.0, 50.0};
    if (!out_dir.empty()) std::filesystem::create_directories(out_dir);

    std::printf("fs = %.0f Hz\n\n", fs);
    std::printf("| scenario | signal | retune ms | frames | median c | p95 c | within 10c | >50c | octave | HF dB | jumps/s | ns/sample |\n");
    std::printf("|---|---|---|---|---|---|---|---|---|---|---|---|\n");

    for (const auto& sc : builtin_scenarios()) {
        const SynthSignal sig = synth_voice(sc.segments, fs, 1);
        const double dur = sig.audio.size() / fs;
        const PitchStats in_stats = score_pitch(sig.audio, sig, 0);
        const double in_hf = hf_energy_db(sig.audio, sig, 0);
        std::printf("| %s | input | – | %s | %.1f | – | – |\n", sc.name.c_str(),
                    format_stats(in_stats).c_str(), in_hf);
        if (!out_dir.empty())
            write_wav(out_dir + "/" + sc.name + "_in.wav",
                      Audio{static_cast<int>(fs), 1, sig.audio});

        for (double rt : retunes) {
            Tuner tuner;
            tuner.prepare(fs);
            TunerParams p;
            p.tonic = sc.tonic;
            p.scale = static_cast<ScaleType>(sc.scale);
            p.retune_ms = rt;
            tuner.set_params(p);
            std::vector<float> out(sig.audio.size());
            const auto t0 = std::chrono::steady_clock::now();
            tuner.process(sig.audio.data(), out.data(), out.size());
            const auto t1 = std::chrono::steady_clock::now();
            const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / out.size();
            const int lat = tuner.latency_samples();
            const PitchStats st = score_pitch(out, sig, lat);
            const double hf = hf_energy_db(out, sig, lat);
            std::printf("| %s | tuned | %.0f | %s | %.1f | %.1f | %.0f |\n", sc.name.c_str(), rt,
                        format_stats(st).c_str(), hf, tuner.jump_count() / dur, ns);
            if (!out_dir.empty()) {
                char name[256];
                std::snprintf(name, sizeof name, "%s/%s_retune%.0f.wav", out_dir.c_str(),
                              sc.name.c_str(), rt);
                write_wav(name, Audio{static_cast<int>(fs), 1, out});
            }
        }
    }
    return 0;
}
