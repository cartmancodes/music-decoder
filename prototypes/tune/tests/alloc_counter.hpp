// Counts heap allocations in the test binary (global operator new is
// replaced in test_main.cpp). Used to prove process() never allocates.
#pragma once
#include <atomic>
#include <cstddef>

namespace tune_test {
extern std::atomic<std::size_t> g_allocations;
struct AllocScope {
    std::size_t start = g_allocations.load();
    std::size_t count() const { return g_allocations.load() - start; }
};
}  // namespace tune_test
