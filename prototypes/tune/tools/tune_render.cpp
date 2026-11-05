// Render a WAV file through the Track-mode tuner (mono; stereo input is
// summed). Writes the tuned WAV and, optionally, a CSV pitch trace.
//
//   tune_render IN.wav OUT.wav [--key A] [--scale minor] [--retune 0]
//               [--amount 1] [--low 70] [--high 800] [--trace trace.csv]
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>

#include "tune/scale.hpp"
#include "tune/tuner.hpp"
#include "tune/wav.hpp"

using namespace tune;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage: %s IN.wav OUT.wav [--key A] [--scale minor] [--retune MS] "
                     "[--amount 0..1] [--low HZ] [--high HZ] [--transpose ST]\n"
                     "       [--formant auto|preserve|follow] [--humanize 0..1] [--flex 0..1]\n"
                     "       [--vibrato 0..2] [--trace FILE.csv]\n",
                     argv[0]);
        return 2;
    }
    const std::string in_path = argv[1], out_path = argv[2];
    TunerParams p;
    p.tonic = 9;
    p.scale = ScaleType::NaturalMinor;
    DetectorConfig det;
    std::string trace_path;
    for (int i = 3; i + 1 < argc; i += 2) {
        const std::string k = argv[i], v = argv[i + 1];
        if (k == "--key") {
            auto t = parse_tonic(v);
            if (!t) { std::fprintf(stderr, "bad key %s\n", v.c_str()); return 2; }
            p.tonic = *t;
        } else if (k == "--scale") {
            auto s = parse_scale(v);
            if (!s) { std::fprintf(stderr, "bad scale %s\n", v.c_str()); return 2; }
            p.scale = *s;
        } else if (k == "--retune") p.retune_ms = std::stod(v);
        else if (k == "--amount") p.amount = std::stod(v);
        else if (k == "--low") det.min_hz = std::stod(v);
        else if (k == "--high") det.max_hz = std::stod(v);
        else if (k == "--trace") trace_path = v;
        else if (k == "--transpose") p.transpose_semitones = std::stod(v);
        else if (k == "--humanize") p.humanize = std::stod(v);
        else if (k == "--flex") p.flex = std::stod(v);
        else if (k == "--vibrato") p.vibrato = std::stod(v);
        else if (k == "--formant") {
            if (v == "auto") p.formant_mode = FormantMode::Auto;
            else if (v == "preserve") p.formant_mode = FormantMode::Preserve;
            else if (v == "follow") p.formant_mode = FormantMode::Follow;
            else { std::fprintf(stderr, "bad --formant %s\n", v.c_str()); return 2; }
        }
        else { std::fprintf(stderr, "unknown option %s\n", k.c_str()); return 2; }
    }
    try {
        const Audio in = read_wav(in_path);
        const std::vector<float> mono = to_mono(in);
        Tuner tuner;
        tuner.prepare(in.sample_rate, det);
        tuner.set_params(p);
        std::vector<float> out(mono.size());
        std::ofstream trace;
        if (!trace_path.empty()) {
            trace.open(trace_path);
            trace << "t,f0_in,target_midi,correction_cents,delay_samples\n";
        }
        const std::size_t trace_every = static_cast<std::size_t>(in.sample_rate / 200);  // 200 Hz
        for (std::size_t i = 0; i < mono.size(); ++i) {
            out[i] = tuner.process_sample(mono[i]);
            if (trace && i % trace_every == 0) {
                const TuneFrame& f = tuner.last_frame();
                trace << f.t << ',' << f.f0_in << ',' << f.target_midi << ',' << f.correction
                      << ',' << f.delay << '\n';
            }
        }
        write_wav(out_path, Audio{in.sample_rate, 1, out});
        std::printf("rendered %zu samples at %d Hz, latency %d samples (%.2f ms), %lld splices\n",
                    out.size(), in.sample_rate, tuner.latency_samples(),
                    1000.0 * tuner.latency_samples() / in.sample_rate,
                    static_cast<long long>(tuner.jump_count()));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
