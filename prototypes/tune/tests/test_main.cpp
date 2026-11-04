#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <infra/doctest.hpp>

#include <cstdlib>
#include <new>

#include "alloc_counter.hpp"

// GCC flags replaced global new/delete backed by malloc/free as "mismatched"
// once it inlines them; the pairing here is correct by construction.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"
#endif

namespace tune_test {
std::atomic<std::size_t> g_allocations{0};
}

void* operator new(std::size_t n) {
    tune_test::g_allocations.fetch_add(1, std::memory_order_relaxed);
    if (void* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void* operator new[](std::size_t n) {
    tune_test::g_allocations.fetch_add(1, std::memory_order_relaxed);
    if (void* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
