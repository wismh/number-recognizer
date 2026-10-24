#pragma once

#include "model/labeled_images.h"
#include "model/parallel.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace game::test {

// A directory that exists for the lifetime of the object.
class TempDir {
public:
    TempDir() {
        static std::uint32_t counter = 0;
        path_ = std::filesystem::temp_directory_path()
                / ("recognizer_tests_" + std::to_string(std::random_device{}()) + "_" + std::to_string(counter++));
        std::filesystem::create_directories(path_);
    }
    ~TempDir() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }
    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;

    [[nodiscard]] const std::filesystem::path& path() const { return path_; }
    [[nodiscard]] std::filesystem::path operator/(const std::string& name) const { return path_ / name; }

private:
    std::filesystem::path path_;
};

inline void write_bytes(const std::filesystem::path& path, const std::vector<std::uint8_t>& bytes) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

template<typename T>
void append(std::vector<std::uint8_t>& bytes, const T& value) {
    const auto* raw = reinterpret_cast<const std::uint8_t*>(&value);
    bytes.insert(bytes.end(), raw, raw + sizeof(T));
}

inline void append_be32(std::vector<std::uint8_t>& bytes, std::uint32_t value) {
    for (int shift = 24; shift >= 0; shift -= 8) {
        bytes.push_back(static_cast<std::uint8_t>(value >> shift));
    }
}

// A filled square of `side` pixels with its top-left corner at (x, y).
inline Pixels square(int x, int y, int side, std::uint8_t value = 255) {
    Pixels pixels{};
    for (int row = y; row < y + side; ++row) {
        for (int col = x; col < x + side; ++col) {
            pixels[static_cast<std::size_t>(row * 28 + col)] = value;
        }
    }
    return pixels;
}

// Ten easy classes: class d is a square at its own place, plus a little noise. Any model can learn them.
inline LabeledImages synthetic_set(std::size_t per_class, std::uint32_t seed) {
    std::mt19937 rng{seed};
    std::uniform_int_distribution<int> noise(0, 40);
    LabeledImages data;
    for (std::size_t i = 0; i < per_class; ++i) {
        for (std::size_t digit = 0; digit < kNumClasses; ++digit) {
            Pixels pixels = square(static_cast<int>(digit % 5) * 5 + 1, static_cast<int>(digit / 5) * 12 + 3, 5);
            for (auto& pixel : pixels) {
                pixel = static_cast<std::uint8_t>(std::min(255, pixel + noise(rng)));
            }
            data.add(pixels, static_cast<std::uint8_t>(digit));
        }
    }
    return data;
}

}
