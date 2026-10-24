#pragma once

#include "model/mlp.h"

#include <atomic>
#include <optional>

namespace game {

// What the model "sees" as the ideal `digit`: an image found by gradient ascent on the model's confidence in that
// digit, starting from faint noise and keeping every pixel in 0..1. Pixels that do not help the digit fade out.
// Reads the model only, so it can run on any thread. Returns nullopt when `cancel` was set.
[[nodiscard]] std::optional<Pixels> dream_digit(const Mlp& model, std::size_t digit, const std::atomic<bool>& cancel);

// The same for one neuron: the image it fires at hardest. `layer` counts from the first layer, output layer included
// (then the neuron is a digit's raw score). Returns nullopt when `cancel` was set.
[[nodiscard]] std::optional<Pixels> dream_neuron(const Mlp& model, std::size_t layer, std::size_t neuron,
        const std::atomic<bool>& cancel);

}
