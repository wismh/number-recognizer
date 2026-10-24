#pragma once

#include "view/picture_paint.h"

#include <engine/ui/view_model.h>

#include <array>

namespace game {

// One row of the neuron grid, kCells pictures wide. Like the dataset grid it is an ItemsControl of these, so only the
// rows on screen are built into elements.
class NeuronRow final : public engine::ui::ViewModel {
public:
    static constexpr std::size_t kCells = 12;

    // `used` cells are bound to neurons; the others at the row's end stay empty and cannot be picked.
    explicit NeuronRow(std::size_t used);

    [[nodiscard]] PicturePaint& cell(std::size_t column) { return cells_[column]; }
    [[nodiscard]] engine::ui::RelayCommand& pick(std::size_t column) { return picks_[column]; }

private:
    std::array<PicturePaint, kCells> cells_;
    std::array<engine::ui::RelayCommand, kCells> picks_;
};

}
