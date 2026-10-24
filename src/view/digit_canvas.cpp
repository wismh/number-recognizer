#include "view/digit_canvas.h"

#include <algorithm>
#include <cmath>

namespace game {

void DigitCanvas::paint(engine::ui::IDrawList& list, const engine::render::Rect& content) {
    list.fill_rect(content, {0.0f, 0.0f, 0.0f, 1.0f});

    constexpr int kSide = static_cast<int>(kImageWidth);
    const float cell_w = content.w / kSide;
    const float cell_h = content.h / kSide;
    for (int y = 0; y < kSide; ++y) {
        for (int x = 0; x < kSide; ++x) {
            const float value = static_cast<float>(pixels_[static_cast<std::size_t>(y * kSide + x)]) / 255.0f;
            if (value > 0.0f) {
                list.fill_rect({content.x + x * cell_w, content.y + y * cell_h, cell_w, cell_h},
                        {value, value, value, 1.0f});
            }
        }
    }
    list.stroke_rect(content, {0.58f, 0.64f, 0.72f, 1.0f}, 2.0f);
}

void DigitCanvas::clear() {
    pixels_.fill(0);
    ++version_;
}

void DigitCanvas::stamp(float nx, float ny) {
    constexpr int kSide = static_cast<int>(kImageWidth);
    // Pointer position in pixel-centre coordinates (pixel i is centred on i), so the brush can sit between pixels.
    const float cx = std::clamp(nx * kSide - 0.5f, 0.0f, kSide - 1.0f);
    const float cy = std::clamp(ny * kSide - 0.5f, 0.0f, kSide - 1.0f);
    const int first_x = static_cast<int>(std::floor(cx)) - 1;
    const int first_y = static_cast<int>(std::floor(cy)) - 1;
    for (int y = first_y; y <= first_y + 3; ++y) {
        for (int x = first_x; x <= first_x + 3; ++x) {
            if (x < 0 || x >= kSide || y < 0 || y >= kSide) {
                continue;
            }
            const float dx = static_cast<float>(x) - cx;
            const float dy = static_cast<float>(y) - cy;
            const float ink = 255.0f * std::exp(-(dx * dx + dy * dy) / (2.0f * kBrushSigma * kBrushSigma));
            std::uint8_t& pixel = pixels_[static_cast<std::size_t>(y * kSide + x)];
            pixel = std::max(pixel, static_cast<std::uint8_t>(std::lround(ink)));
        }
    }
    ++version_;
}

}
