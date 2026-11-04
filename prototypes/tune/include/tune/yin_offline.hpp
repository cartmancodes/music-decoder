// Offline YIN pitch estimator (de Cheveigné & Kawahara 2002), used only as an
// independent judge of the tuner's output pitch. Non-causal, slow, accurate.
#pragma once

#include <vector>

namespace tune {

struct PitchFrame {
    double t = 0.0;        // frame centre, seconds
    double hz = 0.0;       // 0 = unvoiced
    double aperiodicity = 1.0;
};

std::vector<PitchFrame> yin_track(const std::vector<float>& x, double fs, double min_hz = 60.0,
                                  double max_hz = 1000.0, double hop_s = 0.005,
                                  double threshold = 0.12);

}  // namespace tune
