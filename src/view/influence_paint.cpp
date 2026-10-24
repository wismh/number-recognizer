#include "view/influence_paint.h"

#include <engine/builtin_ids.h>

#include <algorithm>
#include <cmath>
#include <format>

namespace game {

namespace {

constexpr float kLabelHeight = 16.0f;
constexpr float kBarGap = 4.0f;

}

void InfluencePaint::paint(engine::ui::IDrawList& list, const engine::render::Rect& content) {
    float strongest = 0.0f;
    for (const float value : influence_) {
        strongest = std::max(strongest, std::abs(value));
    }

    const float plot_height = content.h - kLabelHeight;
    const float axis = content.y + plot_height * 0.5f;
    const float column = content.w / static_cast<float>(kNumClasses);
    const float half = plot_height * 0.5f - 2.0f;

    list.fill_rect({content.x, content.y, content.w, plot_height}, {0.97f, 0.98f, 0.99f, 1.0f});
    list.set_font(engine::builtin::font_ui, 11.0f);
    for (std::size_t digit = 0; digit < kNumClasses; ++digit) {
        const float left = content.x + static_cast<float>(digit) * column;
        const float value = influence_[digit];
        if (strongest > 0.0f && value != 0.0f) {
            const float height = std::max(1.0f, std::abs(value) / strongest * half);
            const bool up = value > 0.0f;
            list.fill_rect({left + kBarGap * 0.5f, up ? axis - height : axis, column - kBarGap, height},
                    up ? glm::vec4{0.86f, 0.27f, 0.17f, 1.0f} : glm::vec4{0.22f, 0.45f, 0.9f, 1.0f});
        }
        list.text(std::format("{}", digit), {left + column * 0.5f - 3.0f, content.y + plot_height + 2.0f},
                {0.2f, 0.25f, 0.33f, 1.0f});
    }
    list.fill_rect({content.x, axis, content.w, 1.0f}, {0.58f, 0.64f, 0.72f, 1.0f});
}

void InfluencePaint::set(const Probabilities& influence) {
    influence_ = influence;
    ++version_;
}

}
