#include "view/probability_paint.h"

#include <engine/builtin_ids.h>

#include <format>

namespace game {

void ProbabilityPaint::paint(engine::ui::IDrawList& list, const engine::render::Rect& content) {
    constexpr float kDigitWidth = 20.0f;
    constexpr float kPercentWidth = 52.0f;
    constexpr float kBarHeight = 18.0f;
    constexpr float kGap = 8.0f;

    const glm::vec4 label_color{0.12f, 0.16f, 0.23f, 1.0f};
    const glm::vec4 percent_color{0.28f, 0.33f, 0.41f, 1.0f};
    const glm::vec4 track_color{0.89f, 0.91f, 0.94f, 1.0f};
    const glm::vec4 strong_color{0.15f, 0.39f, 0.92f, 1.0f};
    const glm::vec4 weak_color{0.39f, 0.45f, 0.55f, 1.0f};

    const float bar_x = content.x + kDigitWidth + kGap;
    const float bar_width = content.w - kDigitWidth - kPercentWidth - 2.0f * kGap;

    list.set_font(engine::builtin::font_ui, 16.0f);
    for (std::size_t digit = 0; digit < kNumClasses; ++digit) {
        const float top = content.y + static_cast<float>(digit) * kRowHeight;
        const float bar_y = top + (kRowHeight - kBarHeight) * 0.5f;
        const float probability = values_[digit];

        list.text(std::format("{}", digit), {content.x, top + 7.0f}, label_color);
        list.fill_rect({bar_x, bar_y, bar_width, kBarHeight}, track_color, 4.0f);
        if (probability > 0.0f) {
            list.fill_rect({bar_x, bar_y, bar_width * probability, kBarHeight},
                    probability > 0.5f ? strong_color : weak_color, 4.0f);
        }
        list.text(probability > 0.001f ? std::format("{:.1f}%", probability * 100.0f) : "0%",
                {bar_x + bar_width + kGap, top + 7.0f}, percent_color);
    }
}

void ProbabilityPaint::set(const Probabilities& probabilities) {
    values_ = probabilities;
    ++version_;
}

void ProbabilityPaint::reset() {
    values_.fill(0.0f);
    ++version_;
}

}
