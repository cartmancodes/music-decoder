#include <infra/doctest.hpp>

#include <algorithm>
#include <initializer_list>

#include "tune/scale.hpp"

using namespace tune;

TEST_CASE("scale masks") {
    const Scale c_major = Scale::make(0, ScaleType::Major);
    for (int pc : {0, 2, 4, 5, 7, 9, 11}) CHECK(c_major.allows(pc));
    for (int pc : {1, 3, 6, 8, 10}) CHECK_FALSE(c_major.allows(pc));
    // A natural minor has the same pitch set as C major (relative keys).
    CHECK(Scale::make(9, ScaleType::NaturalMinor).mask == c_major.mask);
    const Scale a_harm = Scale::make(9, ScaleType::HarmonicMinor);
    CHECK(a_harm.allows(8));  // G#
    CHECK_FALSE(a_harm.allows(7));
    CHECK(Scale::make(5, ScaleType::Chromatic).mask == 0x0FFF);
    CHECK(c_major.allows(-1));  // negative pitch classes wrap (B)
}

TEST_CASE("key and scale parsing") {
    CHECK(parse_tonic("C") == 0);
    CHECK(parse_tonic("C#") == 1);
    CHECK(parse_tonic("Db") == 1);
    CHECK(parse_tonic("a") == 9);
    CHECK(parse_tonic("Cb") == 11);
    CHECK_FALSE(parse_tonic("H").has_value());
    CHECK_FALSE(parse_tonic("C##").has_value());
    CHECK(parse_scale("minor") == ScaleType::NaturalMinor);
    CHECK(parse_scale("blues") == ScaleType::Blues);
    CHECK_FALSE(parse_scale("dorian").has_value());
}

TEST_CASE("nearest allowed note") {
    const Scale c_major = Scale::make(0, ScaleType::Major);
    CHECK(nearest_allowed(c_major, 60.3) == 60);
    CHECK(nearest_allowed(c_major, 61.0) == 60);  // C# -> tie, first found (C) wins
    CHECK(nearest_allowed(c_major, 61.2) == 62);
    CHECK(nearest_allowed(c_major, 65.6) == 65);  // F# region -> F (closer than G)
}

TEST_CASE("note decider hysteresis does not flip near a boundary (FR-TUNE-3)") {
    NoteDecider d;
    d.set_scale(Scale::make(0, ScaleType::Chromatic));
    d.set_hysteresis_cents(15.0);
    CHECK(d.decide(60.0) == 60);
    // Wander +/-40 cents around the 60/61 boundary at 60.5: must stay on 60
    // until the input is clearly closer to 61 (by more than 15 cents).
    int flips = 0, prev = 60;
    for (double m : {60.40, 60.55, 60.45, 60.52, 60.48, 60.56, 60.44}) {
        const int t = d.decide(m);
        if (t != prev) ++flips;
        prev = t;
    }
    CHECK(flips == 0);
    // 60.62 is 38 cents from 61 and 62 cents from 60: closer by 24 cents,
    // more than the 15-cent hysteresis, so the decider moves to 61.
    CHECK(d.decide(60.62) == 61);
}
