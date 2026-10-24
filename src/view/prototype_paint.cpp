#include "view/prototype_paint.h"

#include <engine/builtin_ids.h>

#include <format>

namespace game {

void PrototypePaint::paint(engine::ui::IDrawList& list, const engine::render::Rect& content) {
    constexpr int kSide = 28;
    constexpr float kPixel = kImageSide / kSide;
    const glm::vec4 label_color{0.12f, 0.16f, 0.23f, 1.0f};
    const glm::vec4 empty_color{0.0f, 0.0f, 0.0f, 1.0f};

    list.set_font(engine::builtin::font_ui, 14.0f);
    for (std::size_t digit = 0; digit < kNumClasses; ++digit) {
        const float left = content.x + static_cast<float>(digit % kColumns) * kCellWidth;
        const float top = content.y + static_cast<float>(digit / kColumns) * kCellHeight;
        const float image_left = left + (kCellWidth - kImageSide) * 0.5f;

        list.text(std::format("{}", digit), {left + kCellWidth * 0.5f - 4.0f, top}, label_color);
        const float image_top = top + 14.0f;
        list.fill_rect({image_left, image_top, kImageSide, kImageSide}, empty_color);
        if (!present_[digit]) {
            continue;
        }
        for (int y = 0; y < kSide; ++y) {
            for (int x = 0; x < kSide; ++x) {
                const float value = static_cast<float>(images_[digit][static_cast<std::size_t>(y * kSide + x)]) / 255.0f;
                if (value > 0.0f) {
                    list.fill_rect({image_left + x * kPixel, image_top + y * kPixel, kPixel, kPixel},
                            {value, value, value, 1.0f});
                }
            }
        }
    }
}

void PrototypePaint::set(std::size_t digit, const Pixels& pixels) {
    images_[digit] = pixels;
    present_[digit] = true;
    ++version_;
}

void PrototypePaint::reset() {
    present_.fill(false);
    ++version_;
}

}
