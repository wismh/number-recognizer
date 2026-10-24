#include "view/neuron_row.h"

#include <format>

namespace game {

NeuronRow::NeuronRow(std::size_t used) {
    for (std::size_t column = 0; column < kCells; ++column) {
        picks_[column].set_can_execute(column < used);
        if (column >= used) {
            cells_[column].vacate();
        }
        paint(engine::ui::intern(std::format("cell{}", column)), cells_[column]);
        command(engine::ui::intern(std::format("pick{}", column)), picks_[column]);
    }
}

}
