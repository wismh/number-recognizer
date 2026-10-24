#pragma once

#include "model/mlp.h"

namespace game {

// The weights of a neuron that reads pixels (a neuron of layer 0), laid out as the 28x28 image they are applied to,
// scaled so that its largest weight, in either direction, is +-127. Positive weights are pixels that excite the
// neuron, negative ones pixels that inhibit it.
[[nodiscard]] SignedPixels weight_picture(const Mlp& model, std::size_t layer, std::size_t neuron);

}
