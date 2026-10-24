#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

namespace game {

inline constexpr std::size_t kImageWidth = 28;
inline constexpr std::size_t kImageSize = kImageWidth * kImageWidth;
inline constexpr std::size_t kNumClasses = 10;

using Probabilities = std::array<float, kNumClasses>;
using Pixels = std::array<std::uint8_t, kImageSize>;
// A picture with a sign: -127 is the strongest "against", 127 the strongest "for", 0 is neutral.
using SignedPixels = std::array<std::int8_t, kImageSize>;

[[nodiscard]] inline bool is_blank(const Pixels& pixels) {
    return std::ranges::all_of(pixels, [](std::uint8_t value) { return value == 0; });
}

}
