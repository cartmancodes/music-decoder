#include <infra/doctest.hpp>

#include <algorithm>
#include <initializer_list>

#include <cmath>

#include "tune/correction.hpp"

using namespace tune;

TEST_CASE("retune 0 is a step (FR-TUNE-5 precondition)") {
    RetuneSmoother s;
    s.prepare(48000.0);
    s.set_retune_ms(0.0);
    s.reset(0.0);
    CHECK(s.process(100.0) == doctest::Approx(100.0));
}

TEST_CASE("retune time is the time to ~90% of a step, no overshoot") {
    for (double ms : {10.0, 20.0, 50.0, 200.0}) {
        RetuneSmoother s;
        s.prepare(48000.0);
        s.set_retune_ms(ms);
        s.reset(0.0);
        int n90 = -1;
        double peak = 0.0;
        for (int i = 0; i < 48000; ++i) {
            const double y = s.process(1.0);
            peak = std::max(peak, y);
            if (n90 < 0 && y >= 0.9) n90 = i;
        }
        const double t90_ms = 1000.0 * n90 / 48000.0;
        CHECK(t90_ms == doctest::Approx(ms).epsilon(0.05));
        CHECK(peak <= 1.0 + 1e-12);
    }
}
