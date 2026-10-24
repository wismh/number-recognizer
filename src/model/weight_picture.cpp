#include "model/weight_picture.h"

#include <algorithm>
#include <cmath>

namespace game {

SignedPixels weight_picture(const Mlp& model, std::size_t layer, std::size_t neuron) {
    const std::span<const float> weights = model.incoming_weights(layer, neuron);
    float largest = 0.0f;
    for (const float weight : weights) {
        largest = std::max(largest, std::abs(weight));
    }
    SignedPixels picture{};
    if (largest > 0.0f) {
        for (std::size_t i = 0; i < kImageSize; ++i) {
            picture[i] = static_cast<std::int8_t>(std::lround(weights[i] / largest * 127.0f));
        }
    }
    return picture;
}

}
