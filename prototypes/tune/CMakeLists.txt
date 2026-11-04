# VOX tuner spike (milestone M1): offline C++ test bed for the Track-mode
# tuner. Not shipped; migrates into the rapsuite monorepo (libs/rapcore/pitch)
# once that repo exists. See docs/superpowers/plans/2026-11-04-vox-p0-p1.md.
cmake_minimum_required(VERSION 3.25)
project(tune_spike CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

include(FetchContent)
# Pinned, MIT-licensed (checked 2026-11-04: q LICENSE file; infra per-file
# headers "Distributed under the MIT License"). Header-only use: we only add
# include paths, never their CMake projects (SOURCE_SUBDIR points nowhere).
FetchContent_Declare(cycfi_q
  GIT_REPOSITORY https://github.com/cycfi/q.git
  GIT_TAG efe26114f39e4a78bd67c1456a25487d356fb910
  SOURCE_SUBDIR _none)
FetchContent_Declare(cycfi_infra
  GIT_REPOSITORY https://github.com/cycfi/infra.git
  GIT_TAG f78bd0815c4d4e852bdd89711217d7cef0e8575f
  SOURCE_SUBDIR _none)
FetchContent_MakeAvailable(cycfi_q cycfi_infra)

add_library(tune STATIC src/wav.cpp src/synth.cpp src/yin_offline.cpp src/metrics.cpp)
target_include_directories(tune PUBLIC include)
target_include_directories(tune SYSTEM PUBLIC
  ${cycfi_q_SOURCE_DIR}/q_lib/include ${cycfi_infra_SOURCE_DIR}/include)
target_compile_options(tune PUBLIC -Wall -Wextra -Wno-unused-parameter)

add_executable(tune_render tools/tune_render.cpp)
target_link_libraries(tune_render PRIVATE tune)

add_executable(tune_eval tools/tune_eval.cpp)
target_link_libraries(tune_eval PRIVATE tune)

enable_testing()
add_executable(tune_tests tests/test_main.cpp tests/test_scale.cpp
  tests/test_correction.cpp tests/test_shifter.cpp tests/test_tuner.cpp)
target_link_libraries(tune_tests PRIVATE tune)
add_test(NAME tune_tests COMMAND tune_tests)
