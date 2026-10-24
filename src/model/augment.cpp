#include "model/augment.h"

#include <algorithm>
#include <cmath>

namespace game {

namespace {

constexpr int kSide = 28;

struct Bounds {
    int min_x = kSide;
    int max_x = -1;
    int min_y = kSide;
    int max_y = -1;

    [[nodiscard]] bool empty() const { return max_x < 0; }
};

Bounds bounds_of(std::span<const std::uint8_t> image) {
    Bounds bounds;
    for (int y = 0; y < kSide; ++y) {
        for (int x = 0; x < kSide; ++x) {
            if (image[static_cast<std::size_t>(y * kSide + x)] > 0) {
                bounds.min_x = std::min(bounds.min_x, x);
                bounds.max_x = std::max(bounds.max_x, x);
                bounds.min_y = std::min(bounds.min_y, y);
                bounds.max_y = std::max(bounds.max_y, y);
            }
        }
    }
    return bounds;
}

float sample(std::span<const std::uint8_t> image, int x, int y) {
    if (x < 0 || x >= kSide || y < 0 || y >= kSide) {
        return 0.0f;
    }
    return static_cast<float>(image[static_cast<std::size_t>(y * kSide + x)]);
}

}

void random_shift(std::span<const std::uint8_t> image, std::uint64_t random, Pixels& out) {
    const Bounds b = bounds_of(image);
    out.fill(0);
    if (b.empty()) {
        return;
    }

    // dx in [-min_x, kSide - 1 - max_x], inclusive.
    const int span_x = kSide - (b.max_x - b.min_x);
    const int span_y = kSide - (b.max_y - b.min_y);
    const int dx = static_cast<int>((random & 0xffffffffULL) % static_cast<std::uint64_t>(span_x)) - b.min_x;
    const int dy = static_cast<int>((random >> 32) % static_cast<std::uint64_t>(span_y)) - b.min_y;

    for (int y = b.min_y; y <= b.max_y; ++y) {
        for (int x = b.min_x; x <= b.max_x; ++x) {
            out[static_cast<std::size_t>((y + dy) * kSide + x + dx)] = image[static_cast<std::size_t>(y * kSide + x)];
        }
    }
}

void random_scale(std::span<const std::uint8_t> image, std::uint64_t random, Pixels& out) {
    const Bounds b = bounds_of(image);
    out.fill(0);
    if (b.empty()) {
        std::ranges::copy(image, out.begin());
        return;
    }

    const float width = static_cast<float>(b.max_x - b.min_x + 1);
    const float height = static_cast<float>(b.max_y - b.min_y + 1);
    const float unit = static_cast<float>(random & 0xffffffULL) / static_cast<float>(0xffffff);
    const float factor = std::min(kMinScale + (kMaxScale - kMinScale) * unit,
            static_cast<float>(kSide) / std::max(width, height));

    // Centre of the box in pixel-edge coordinates (pixel i covers [i, i + 1)), then moved inward if the scaled box
    // would cross the frame.
    const float cx = static_cast<float>(b.min_x + b.max_x + 1) * 0.5f;
    const float cy = static_cast<float>(b.min_y + b.max_y + 1) * 0.5f;
    const float half_w = width * factor * 0.5f;
    const float half_h = height * factor * 0.5f;
    const float ncx = std::clamp(cx, half_w, static_cast<float>(kSide) - half_w);
    const float ncy = std::clamp(cy, half_h, static_cast<float>(kSide) - half_h);

    for (int y = 0; y < kSide; ++y) {
        for (int x = 0; x < kSide; ++x) {
            // Source position (in pixel-index coordinates) that lands on the centre of this output pixel.
            const float sx = cx + (static_cast<float>(x) + 0.5f - ncx) / factor - 0.5f;
            const float sy = cy + (static_cast<float>(y) + 0.5f - ncy) / factor - 0.5f;
            const int x0 = static_cast<int>(std::floor(sx));
            const int y0 = static_cast<int>(std::floor(sy));
            const float fx = sx - static_cast<float>(x0);
            const float fy = sy - static_cast<float>(y0);
            const float top = sample(image, x0, y0) * (1.0f - fx) + sample(image, x0 + 1, y0) * fx;
            const float bottom = sample(image, x0, y0 + 1) * (1.0f - fx) + sample(image, x0 + 1, y0 + 1) * fx;
            const float value = top * (1.0f - fy) + bottom * fy;
            out[static_cast<std::size_t>(y * kSide + x)] = static_cast<std::uint8_t>(std::lround(value));
        }
    }
}

std::span<const std::uint8_t> augmented(std::span<const std::uint8_t> image, Augmentation augmentation,
        std::uint64_t random, Pixels& scratch_a, Pixels& scratch_b) {
    std::span<const std::uint8_t> current = image;
    if (augmentation.scale) {
        random_scale(current, mix_bits(random), scratch_a);
        current = scratch_a;
    }
    if (augmentation.shift) {
        Pixels& target = augmentation.scale ? scratch_b : scratch_a;
        random_shift(current, mix_bits(random ^ 0x5bd1e995ULL), target);
        current = target;
    }
    return current;
}

}
