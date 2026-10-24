#pragma once

#include "model/digits.h"

#include <engine/ui/paint.h>

namespace game {

// A 28x28 picture drawn into whatever element it is bound to: a neuron in the grid, the big one beside it, the player's
// drawing. A cell of the grid also carries its number, a bar for how strongly the neuron reacts, and a frame.
class PicturePaint final : public engine::ui::IPaint {
public:
    // Gray: 0..127 is black to white. Diverging: positive is red, negative is blue, brighter is stronger.
    enum class Palette { Gray, Diverging };

    static constexpr int kNoLabel = -1;

    void paint(engine::ui::IDrawList& list, const engine::render::Rect& content) override;
    [[nodiscard]] std::uint64_t version() const noexcept override { return version_; }

    void show(const SignedPixels& picture, Palette palette);
    void show_gray(const Pixels& picture);
    // Nothing to see yet (a dream still being worked out).
    void hide();
    // Paints nothing at all: the unused cell at the end of a grid row.
    void vacate();
    void set_selected(bool selected);
    // 0..1, or negative for no bar. Quantized, so a small change does not redraw the picture.
    void set_meter(float meter);
    void set_label(int label);

private:
    SignedPixels picture_{};
    Palette palette_ = Palette::Gray;
    bool present_ = false;
    bool vacant_ = false;
    bool selected_ = false;
    int meter_steps_ = -1;
    int label_ = kNoLabel;
    std::uint64_t version_ = 0;
};

}
