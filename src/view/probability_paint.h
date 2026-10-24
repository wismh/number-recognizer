#pragma once

#include "model/digits.h"

#include <engine/ui/paint.h>

namespace game {

// Ten horizontal bars, one per digit, with the probability as a percentage.
class ProbabilityPaint final : public engine::ui::IPaint {
public:
    static constexpr float kRowHeight = 34.0f;

    void paint(engine::ui::IDrawList& list, const engine::render::Rect& content) override;
    [[nodiscard]] std::uint64_t version() const noexcept override { return version_; }

    void set(const Probabilities& probabilities);
    void reset();

private:
    Probabilities values_{};
    std::uint64_t version_ = 0;
};

}
