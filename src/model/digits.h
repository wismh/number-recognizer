#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace game {

inline constexpr std::size_t kImageSize = 28 * 28;
inline constexpr std::size_t kNumClasses = 10;

using Probabilities = std::array<float, kNumClasses>;
using Pixels = std::array<std::uint8_t, kImageSize>;
// A picture with a sign: -127 is the strongest "against", 127 the strongest "for", 0 is neutral.
using SignedPixels = std::array<std::int8_t, kImageSize>;

}
