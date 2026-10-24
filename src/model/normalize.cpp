#include "model/normalize.h"

#include <algorithm>
#include <cmath>

namespace game {

namespace {

constexpr int kSide = static_cast<int>(kImageWidth);
constexpr float kDigitSize = 20.0f;
constexpr float kCentre = 14.0f;

float sample(const Pixels& image, int x, int y) {
    if (x < 0 || x >= kSide || y < 0 || y >= kSide) {
        return 0.0f;
    }
    return static_cast<float>(image[static_cast<std::size_t>(y * kSide + x)]);
}

}

Pixels normalize_digit(const Pixels& image) {
    int min_x = kSide;
    int max_x = -1;
    int min_y = kSide;
    int max_y = -1;
    double mass = 0.0;
    double mass_x = 0.0;
    double mass_y = 0.0;
    for (int y = 0; y < kSide; ++y) {
        for (int x = 0; x < kSide; ++x) {
            const double value = image[static_cast<std::size_t>(y * kSide + x)];
            if (value <= 0.0) {
                continue;
            }
            min_x = std::min(min_x, x);
            max_x = std::max(max_x, x);
            min_y = std::min(min_y, y);
            max_y = std::max(max_y, y);
            mass += value;
            mass_x += value * (x + 0.5);
            mass_y += value * (y + 0.5);
        }
    }
    if (mass <= 0.0) {
        return image;
    }

    // Everything below is in pixel-edge coordinates: pixel i covers [i, i + 1).
    const auto com_x = static_cast<float>(mass_x / mass);
    const auto com_y = static_cast<float>(mass_y / mass);
    const float scale = kDigitSize / static_cast<float>(std::max(max_x - min_x + 1, max_y - min_y + 1));

    // Where the box edges land relative to the centre of mass once scaled; the centre of mass goes to the middle of
    // the frame unless that would push the box out.
    const float left = (static_cast<float>(min_x) - com_x) * scale;
    const float right = (static_cast<float>(max_x + 1) - com_x) * scale;
    const float top = (static_cast<float>(min_y) - com_y) * scale;
    const float bottom = (static_cast<float>(max_y + 1) - com_y) * scale;
    const float out_x = std::clamp(kCentre, -left, static_cast<float>(kSide) - right);
    const float out_y = std::clamp(kCentre, -top, static_cast<float>(kSide) - bottom);

    Pixels result{};
    for (int y = 0; y < kSide; ++y) {
        for (int x = 0; x < kSide; ++x) {
            const float sx = com_x + (static_cast<float>(x) + 0.5f - out_x) / scale - 0.5f;
            const float sy = com_y + (static_cast<float>(y) + 0.5f - out_y) / scale - 0.5f;
            const int x0 = static_cast<int>(std::floor(sx));
            const int y0 = static_cast<int>(std::floor(sy));
            const float fx = sx - static_cast<float>(x0);
            const float fy = sy - static_cast<float>(y0);
            const float upper = sample(image, x0, y0) * (1.0f - fx) + sample(image, x0 + 1, y0) * fx;
            const float lower = sample(image, x0, y0 + 1) * (1.0f - fx) + sample(image, x0 + 1, y0 + 1) * fx;
            result[static_cast<std::size_t>(y * kSide + x)] =
                    static_cast<std::uint8_t>(std::lround(upper * (1.0f - fy) + lower * fy));
        }
    }
    return result;
}

}
