#include "tune/wav.hpp"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace tune {
namespace {

std::uint32_t u32(const unsigned char* p) {
    return std::uint32_t(p[0]) | (std::uint32_t(p[1]) << 8) | (std::uint32_t(p[2]) << 16) |
           (std::uint32_t(p[3]) << 24);
}
std::uint16_t u16(const unsigned char* p) {
    return std::uint16_t(p[0] | (p[1] << 8));
}

void put_u32(std::ofstream& f, std::uint32_t v) {
    unsigned char b[4] = {static_cast<unsigned char>(v), static_cast<unsigned char>(v >> 8),
                          static_cast<unsigned char>(v >> 16), static_cast<unsigned char>(v >> 24)};
    f.write(reinterpret_cast<const char*>(b), 4);
}
void put_u16(std::ofstream& f, std::uint16_t v) {
    unsigned char b[2] = {static_cast<unsigned char>(v), static_cast<unsigned char>(v >> 8)};
    f.write(reinterpret_cast<const char*>(b), 2);
}

}  // namespace

Audio read_wav(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(f)), {});
    if (bytes.size() < 12 || std::memcmp(bytes.data(), "RIFF", 4) != 0 ||
        std::memcmp(bytes.data() + 8, "WAVE", 4) != 0)
        throw std::runtime_error("not a RIFF/WAVE file: " + path);

    int format = 0, channels = 0, rate = 0, bits = 0;
    const unsigned char* data = nullptr;
    std::size_t data_len = 0;
    std::size_t pos = 12;
    while (pos + 8 <= bytes.size()) {
        const unsigned char* chunk = bytes.data() + pos;
        std::size_t len = u32(chunk + 4);
        const unsigned char* body = chunk + 8;
        if (pos + 8 + len > bytes.size()) len = bytes.size() - pos - 8;  // tolerate truncation
        if (std::memcmp(chunk, "fmt ", 4) == 0 && len >= 16) {
            format = u16(body);
            channels = u16(body + 2);
            rate = static_cast<int>(u32(body + 4));
            bits = u16(body + 14);
            if (format == 0xFFFE && len >= 26) format = u16(body + 24);  // extensible subformat
        } else if (std::memcmp(chunk, "data", 4) == 0) {
            data = body;
            data_len = len;
        }
        pos += 8 + len + (len & 1);
    }
    if (!data || channels <= 0 || rate <= 0) throw std::runtime_error("missing fmt/data: " + path);

    Audio a;
    a.sample_rate = rate;
    a.channels = channels;
    const int bytes_per = bits / 8;
    if (bytes_per <= 0) throw std::runtime_error("bad bit depth: " + path);
    const std::size_t n = data_len / bytes_per;
    a.data.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        const unsigned char* p = data + i * bytes_per;
        float v = 0.0f;
        if (format == 1 && bits == 16) {
            v = static_cast<std::int16_t>(u16(p)) / 32768.0f;
        } else if (format == 1 && bits == 24) {
            std::int32_t s = (std::int32_t(p[0]) << 8) | (std::int32_t(p[1]) << 16) |
                             (std::int32_t(p[2]) << 24);
            v = static_cast<float>(s >> 8) / 8388608.0f;
        } else if (format == 1 && bits == 32) {
            v = static_cast<float>(static_cast<std::int32_t>(u32(p)) / 2147483648.0);
        } else if (format == 3 && bits == 32) {
            std::uint32_t w = u32(p);
            std::memcpy(&v, &w, 4);
        } else {
            throw std::runtime_error("unsupported WAV format " + std::to_string(format) + "/" +
                                     std::to_string(bits) + ": " + path);
        }
        a.data[i] = v;
    }
    a.data.resize(a.frames() * a.channels);
    return a;
}

void write_wav(const std::string& path, const Audio& a) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot write " + path);
    const std::uint32_t data_len = static_cast<std::uint32_t>(a.data.size() * 4);
    f.write("RIFF", 4);
    put_u32(f, 36 + data_len);
    f.write("WAVEfmt ", 8);
    put_u32(f, 16);
    put_u16(f, 3);  // IEEE float
    put_u16(f, static_cast<std::uint16_t>(a.channels));
    put_u32(f, static_cast<std::uint32_t>(a.sample_rate));
    put_u32(f, static_cast<std::uint32_t>(a.sample_rate * a.channels * 4));
    put_u16(f, static_cast<std::uint16_t>(a.channels * 4));
    put_u16(f, 32);
    f.write("data", 4);
    put_u32(f, data_len);
    for (float v : a.data) {
        std::uint32_t w;
        std::memcpy(&w, &v, 4);
        put_u32(f, w);
    }
    if (!f) throw std::runtime_error("write failed: " + path);
}

std::vector<float> to_mono(const Audio& a) {
    std::vector<float> m(a.frames());
    for (std::size_t i = 0; i < m.size(); ++i) {
        float s = 0.0f;
        for (int c = 0; c < a.channels; ++c) s += a.data[i * a.channels + c];
        m[i] = s / static_cast<float>(a.channels);
    }
    return m;
}

}  // namespace tune
