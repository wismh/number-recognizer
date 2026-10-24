#pragma once

#include "model/digits.h"

#include <engine/ui/paint.h>

namespace game {

// Ten bars, one per digit, around a middle line: red going up for the digits a neuron pushes toward, blue going down
// for the digits it pushes away from. All bars share one scale, so the strongest effect fills the height.
class InfluencePaint final : public engine::ui::IPaint {
public:
    void paint(engine::ui::IDrawList& list, const engine::render::Rect& content) override;
    [[nodiscard]] std::uint64_t version() const noexcept override { return version_; }

    void set(const Probabilities& influence);

private:
    Probabilities influence_{};
    std::uint64_t version_ = 0;
};

}
