#pragma once

#include "model/digits.h"

#include <cstdint>
#include <span>

namespace game {

// Which random changes a training sample goes through.
struct Augmentation {
    bool shift = false;
    bool scale = false;
};

// Copies `image` into `out` moved by a random whole-pixel offset, chosen so that every lit pixel stays inside the
// 28x28 frame: the offset ranges from "touching the left/top edge" to "touching the right/bottom edge". The two
// halves of `random` pick x and y. A blank image is copied as is.
void random_shift(std::span<const std::uint8_t> image, std::uint64_t random, Pixels& out);

// Copies `image` into `out` with the digit scaled about the centre of its bounding box by a random factor between
// kMinScale and kMaxScale (bilinear). The factor is capped so the whole digit still fits in the frame; the box is
// nudged inward if it would stick out. A blank image is copied as is.
inline constexpr float kMinScale = 0.7f;
inline constexpr float kMaxScale = 1.4f;
void random_scale(std::span<const std::uint8_t> image, std::uint64_t random, Pixels& out);

// A well-mixed 64-bit value from `seed`, for deriving one random number per sample without shared generator state.
[[nodiscard]] constexpr std::uint64_t mix_bits(std::uint64_t seed) {
    seed += 0x9e3779b97f4a7c15ULL;
    seed = (seed ^ (seed >> 30)) * 0xbf58476d1ce4e5b9ULL;
    seed = (seed ^ (seed >> 27)) * 0x94d049bb133111ebULL;
    return seed ^ (seed >> 31);
}

// `image` after the enabled changes (scale first, then shift), using `random` for both. Returns `image` itself when
// nothing is enabled, otherwise a view of one of the two scratch buffers.
[[nodiscard]] std::span<const std::uint8_t> augmented(std::span<const std::uint8_t> image, Augmentation augmentation,
        std::uint64_t random, Pixels& scratch_a, Pixels& scratch_b);

}
