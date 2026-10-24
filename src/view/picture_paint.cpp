#include "view/picture_paint.h"

#include <engine/builtin_ids.h>

#include <algorithm>
#include <cmath>
#include <format>

namespace game {

namespace {

constexpr int kSide = 28;
constexpr int kMeterSteps = 16;
constexpr float kMeterHeight = 3.0f;
// Weights this small next to the largest one are left black: they only make the picture noisy.
constexpr float kFaint = 0.04f;

glm::vec4 color_of(std::int8_t value, PicturePaint::Palette palette) {
    const float strength = std::abs(static_cast<float>(value)) / 127.0f;
    if (palette == PicturePaint::Palette::Gray) {
        return {strength, strength, strength, 1.0f};
    }
    // Weights are mostly small and a few are large; the root keeps the small ones visible.
    const float t = std::sqrt(strength);
    return value > 0 ? glm::vec4{t, 0.22f * t, 0.12f * t, 1.0f} : glm::vec4{0.12f * t, 0.45f * t, t, 1.0f};
}

}

void PicturePaint::paint(engine::ui::IDrawList& list, const engine::render::Rect& content) {
    if (vacant_) {
        return;
    }
    list.fill_rect(content, present_ ? glm::vec4{0.0f, 0.0f, 0.0f, 1.0f} : glm::vec4{0.12f, 0.16f, 0.23f, 1.0f});
    if (present_) {
        const float pixel = content.w / kSide;
        for (int y = 0; y < kSide; ++y) {
            for (int x = 0; x < kSide; ++x) {
                const std::int8_t value = picture_[static_cast<std::size_t>(y * kSide + x)];
                if (std::abs(static_cast<float>(value)) / 127.0f >= kFaint) {
                    list.fill_rect({content.x + x * pixel, content.y + y * pixel, pixel, pixel},
                            color_of(value, palette_));
                }
            }
        }
    }

    if (label_ != kNoLabel) {
        list.set_font(engine::builtin::font_ui, 10.0f);
        list.text(std::format("{}", label_), {content.x + 2.0f, content.y}, {0.9f, 0.9f, 0.9f, 0.75f});
    }
    if (meter_steps_ > 0) {
        const float width = content.w * static_cast<float>(meter_steps_) / kMeterSteps;
        list.fill_rect({content.x, content.y + content.h - kMeterHeight, width, kMeterHeight},
                {0.35f, 0.85f, 0.45f, 1.0f});
    }
    if (selected_) {
        list.stroke_rect(content, {1.0f, 0.78f, 0.2f, 1.0f}, 2.0f);
    }
}

void PicturePaint::show(const SignedPixels& picture, Palette palette) {
    picture_ = picture;
    palette_ = palette;
    present_ = true;
    ++version_;
}

void PicturePaint::show_gray(const Pixels& picture) {
    SignedPixels signed_picture;
    for (std::size_t i = 0; i < kImageSize; ++i) {
        signed_picture[i] = static_cast<std::int8_t>(picture[i] >> 1);
    }
    show(signed_picture, Palette::Gray);
}

void PicturePaint::hide() {
    present_ = false;
    ++version_;
}

void PicturePaint::vacate() {
    vacant_ = true;
    ++version_;
}

void PicturePaint::set_selected(bool selected) {
    if (selected_ != selected) {
        selected_ = selected;
        ++version_;
    }
}

void PicturePaint::set_meter(float meter) {
    const int steps = meter < 0.0f ? -1 : static_cast<int>(std::lround(std::min(meter, 1.0f) * kMeterSteps));
    if (meter_steps_ != steps) {
        meter_steps_ = steps;
        ++version_;
    }
}

void PicturePaint::set_label(int label) {
    if (label_ != label) {
        label_ = label;
        ++version_;
    }
}

}
