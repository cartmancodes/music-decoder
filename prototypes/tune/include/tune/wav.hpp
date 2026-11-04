// Minimal WAV I/O for the spike (PCM 16/24/32-bit and 32-bit float,
// including WAVE_FORMAT_EXTENSIBLE). Not real-time code.
#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace tune {

struct Audio {
    int sample_rate = 48000;
    int channels = 1;
    std::vector<float> data;  // interleaved

    std::size_t frames() const { return channels ? data.size() / channels : 0; }
};

// Throws std::runtime_error on unreadable or unsupported files.
Audio read_wav(const std::string& path);

// Writes 32-bit float WAV. Throws std::runtime_error on failure.
void write_wav(const std::string& path, const Audio& audio);

// Average of all channels.
std::vector<float> to_mono(const Audio& audio);

}  // namespace tune
