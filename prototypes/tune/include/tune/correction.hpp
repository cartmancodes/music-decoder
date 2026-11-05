// Retune smoother (spec §6.3). Two cascaded identical one-pole low-passes
// form a critically damped second-order response (no overshoot).
// `retune_ms` is defined as the time to reach ~90% of a step, so each stage's
// time constant is retune_ms / 3.89. retune_ms == 0 means a step change.
#pragma once

#include <cmath>

namespace tune {

class RetuneSmoother {
public:
    void prepare(double sample_rate) {
        fs_ = sample_rate;
        set_retune_ms(retune_ms_);
    }

    void set_retune_ms(double ms) {
        retune_ms_ = ms < 0.0 ? 0.0 : ms;
        if (retune_ms_ <= 0.0) {
            a_ = 0.0;
        } else {
            const double tau_s = (retune_ms_ / 1000.0) / 3.89;  // per stage
            a_ = std::exp(-1.0 / (tau_s * fs_));
        }
    }

    double retune_ms() const { return retune_ms_; }

    void reset(double value = 0.0) { y1_ = y2_ = value; }

    double process(double target) {
        if (a_ == 0.0) {
            y1_ = y2_ = target;
            return target;
        }
        y1_ = target + a_ * (y1_ - target);
        y2_ = y1_ + a_ * (y2_ - y1_);
        return y2_;
    }

    double value() const { return y2_; }

private:
    double fs_ = 48000.0;
    double retune_ms_ = 20.0;
    double a_ = 0.0;
    double y1_ = 0.0, y2_ = 0.0;
};

// One-pole smoother used for fades of the correction toward 0 when unvoiced.
class OnePole {
public:
    void prepare(double sample_rate, double time_ms) {
        a_ = time_ms <= 0.0 ? 0.0 : std::exp(-1.0 / ((time_ms / 1000.0) * sample_rate));
    }
    void reset(double v = 0.0) { y_ = v; }
    double process(double x) {
        y_ = x + a_ * (y_ - x);
        return y_;
    }
    double value() const { return y_; }

private:
    double a_ = 0.0, y_ = 0.0;
};

// Second-order Butterworth low-pass on a control signal (pitch in semitones),
// used to split the detected pitch into a slow "centre" and the vibrato band
// (spec §6.3). reset(v) puts the filter in steady state at v, so a new note
// starts without a transient.
class ControlLowpass {
public:
    void prepare(double fs, double cutoff_hz) {
        const double pi = 3.14159265358979323846;
        const double w = 2.0 * pi * cutoff_hz / fs;
        const double cw = std::cos(w), alpha = std::sin(w) / (2.0 * 0.70710678118654752);
        const double a0 = 1.0 + alpha;
        b0_ = (1.0 - cw) / 2.0 / a0;
        b1_ = (1.0 - cw) / a0;
        b2_ = b0_;
        a1_ = -2.0 * cw / a0;
        a2_ = (1.0 - alpha) / a0;
    }
    void reset(double v) {
        // Transposed direct form II steady state for constant input v (DC gain 1).
        z1_ = v * (1.0 - b0_);
        z2_ = v * (b2_ - a2_);
        y_ = v;
    }
    double process(double x) {
        y_ = b0_ * x + z1_;
        z1_ = b1_ * x - a1_ * y_ + z2_;
        z2_ = b2_ * x - a2_ * y_;
        return y_;
    }
    double value() const { return y_; }

private:
    double b0_ = 1, b1_ = 0, b2_ = 0, a1_ = 0, a2_ = 0, z1_ = 0, z2_ = 0, y_ = 0;
};

}  // namespace tune
