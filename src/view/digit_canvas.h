#pragma once

#include "model/digits.h"

#include <engine/ui/paint.h>

namespace game {

// The 28x28 drawing surface: owns the pixels and paints them scaled to the element it is bound to.
class DigitCanvas final : public engine::ui::IPaint {
public:
    // Soft round brush: ink falls off as a Gaussian of this width (in pixels of the 28x28 grid) from the pointer,
    // which gives MNIST-like strokes about two pixels wide with soft edges.
    static constexpr float kBrushSigma = 0.7f;

    void paint(engine::ui::IDrawList& list, const engine::render::Rect& content) override;
    [[nodiscard]] std::uint64_t version() const noexcept override { return version_; }

    void clear();
    // Adds a soft dot of ink at the normalized position (0..1 on both axes); it never lightens existing ink.
    void stamp(float nx, float ny);

    [[nodiscard]] const Pixels& pixels() const { return pixels_; }

private:
    Pixels pixels_{};
    std::uint64_t version_ = 0;
};

}
