#include <infra/doctest.hpp>

#include <algorithm>
#include <initializer_list>

#include <cmath>
#include <vector>

#include "alloc_counter.hpp"
#include "tune/splice_shifter.hpp"

using namespace tune;

namespace {
constexpr double kPi = 3.14159265358979323846;

std::vector<float> sine(double hz, double fs, std::size_t n, double amp = 0.5) {
    std::vector<float> x(n);
    for (std::size_t i = 0; i < n; ++i) x[i] = static_cast<float>(amp * std::sin(2 * kPi * hz * i / fs));
    return x;
}

// Frequency from positive-going zero crossings with linear interpolation.
double zc_frequency(const std::vector<float>& y, double fs, std::size_t from) {
    double first = -1, last = -1;
    int count = 0;
    for (std::size_t i = from + 1; i < y.size(); ++i) {
        if (y[i - 1] < 0.0f && y[i] >= 0.0f) {
            const double t = (i - 1) + y[i - 1] / (y[i - 1] - y[i]);
            if (first < 0) first = t;
            else ++count;
            last = t;
        }
    }
    return count > 0 ? fs * count / (last - first) : 0.0;
}
}  // namespace

TEST_CASE("ratio 1 is an exact delay of the reported latency (FR-LAT-1)") {
    SpliceShifter s;
    ShifterConfig cfg;
    s.prepare(cfg);
    const int lat = s.latency();
    const auto x = sine(220.0, 48000.0, 4800);
    std::vector<float> y(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) y[i] = s.process(x[i], 1.0, 48000.0 / 220.0);
    for (std::size_t i = lat; i < x.size(); ++i) REQUIRE(y[i] == x[i - lat]);
    for (int i = 0; i < lat; ++i) CHECK(y[i] == 0.0f);
}

TEST_CASE("constant ratio shifts a sine by that ratio without clicks") {
    const double fs = 48000.0;
    for (double semis : {-3.0, -1.0, 0.5, 1.0, 3.0}) {
        const double ratio = std::exp2(semis / 12.0);
        SpliceShifter s;
        s.prepare(ShifterConfig{});
        const double f_in = 220.0;
        const auto x = sine(f_in, fs, 48000);
        std::vector<float> y(x.size());
        for (std::size_t i = 0; i < x.size(); ++i) y[i] = s.process(x[i], ratio, fs / f_in);
        const double f_out = zc_frequency(y, fs, 4800);
        CHECK(f_out == doctest::Approx(f_in * ratio).epsilon(0.003));
        // A sine of amplitude A and frequency f never changes by more than
        // 2*pi*f*A/fs per sample; allow 25% for interpolation and crossfades.
        const double max_step = 2 * kPi * f_out * 0.5 / fs * 1.25;
        double worst = 0.0;
        for (std::size_t i = 4801; i < y.size(); ++i)
            worst = std::max(worst, static_cast<double>(std::abs(y[i] - y[i - 1])));
        CHECK(worst <= max_step);
        CHECK(s.jump_count() > 0);
    }
}

TEST_CASE("unvoiced input returns the head to the rest delay") {
    SpliceShifter s;
    s.prepare(ShifterConfig{});
    const auto x = sine(220.0, 48000.0, 9600);
    for (std::size_t i = 0; i < 4800; ++i) s.process(x[i], 1.06, 48000.0 / 220.0);
    CHECK(s.current_delay() != doctest::Approx(s.latency()));
    for (std::size_t i = 4800; i < 9600; ++i) s.process(x[i], 1.0, 0.0);
    CHECK(s.current_delay() == doctest::Approx(s.latency()));
}

TEST_CASE("delay wander stays within [latency, latency + period + margin]") {
    const double fs = 48000.0, f_in = 110.0, period = fs / f_in;
    for (double ratio : {std::exp2(-2.0 / 12), std::exp2(2.0 / 12)}) {
        SpliceShifter s;
        ShifterConfig cfg;
        s.prepare(cfg);
        const auto x = sine(f_in, fs, 48000);
        double lo = 1e9, hi = -1e9;
        for (std::size_t i = 0; i < x.size(); ++i) {
            s.process(x[i], ratio, period);
            if (i > 4800) {
                lo = std::min(lo, s.current_delay());
                hi = std::max(hi, s.current_delay());
            }
        }
        CHECK(lo >= s.latency() - 0.03 * period - 1);
        CHECK(hi <= s.latency() + period * 1.03 + cfg.crossfade + 2);
    }
}

TEST_CASE("process() does not allocate (NFR-8 proxy)") {
    SpliceShifter s;
    s.prepare(ShifterConfig{});
    const auto x = sine(150.0, 48000.0, 48000);
    tune_test::AllocScope scope;
    float acc = 0.0f;
    for (std::size_t i = 0; i < x.size(); ++i) acc += s.process(x[i], 1.1, 320.0);
    CHECK(scope.count() == 0);
    CHECK(std::isfinite(acc));
}
