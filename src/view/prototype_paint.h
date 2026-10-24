#pragma once

#include "model/digits.h"

#include <engine/ui/paint.h>

namespace game {

// Ten 28x28 pictures in two rows of five, each under its digit: the model's typical example of every digit.
class PrototypePaint final : public engine::ui::IPaint {
public:
    static constexpr float kCellWidth = 48.0f;
    static constexpr float kCellHeight = 56.0f;
    static constexpr float kImageSide = 42.0f;
    static constexpr std::size_t kColumns = 5;

    void paint(engine::ui::IDrawList& list, const engine::render::Rect& content) override;
    [[nodiscard]] std::uint64_t version() const noexcept override { return version_; }

    void set(std::size_t digit, const Pixels& pixels);
    void reset();

private:
    std::array<Pixels, kNumClasses> images_{};
    std::array<bool, kNumClasses> present_{};
    std::uint64_t version_ = 0;
};

}
