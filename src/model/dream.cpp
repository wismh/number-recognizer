#include "model/dream.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace game {

namespace {

constexpr int kDigitSteps = 400;
// A neuron is looked at by many tasks at once and a deep one costs every layer below it per step, so it gets fewer.
constexpr int kNeuronSteps = 200;
constexpr float kFirstStep = 0.05f;
constexpr float kLastStep = 0.005f;
// Per step, every pixel shrinks by this share, which keeps the image to what the digit needs.
constexpr float kDecay = 0.01f;
constexpr float kStartNoise = 0.1f;
// Per step, every pixel moves this far toward the mean of its 3x3 neighbourhood, so strokes come out connected
// instead of as scattered dots.
constexpr float kBlur = 0.15f;

void blur(std::vector<float>& image) {
    constexpr int kSide = 28;
    const std::vector<float> source = image;
    for (int y = 0; y < kSide; ++y) {
        for (int x = 0; x < kSide; ++x) {
            float sum = 0.0f;
            int count = 0;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const int nx = x + dx;
                    const int ny = y + dy;
                    if (nx >= 0 && nx < kSide && ny >= 0 && ny < kSide) {
                        sum += source[static_cast<std::size_t>(ny * kSide + nx)];
                        ++count;
                    }
                }
            }
            float& value = image[static_cast<std::size_t>(y * kSide + x)];
            value += kBlur * (sum / static_cast<float>(count) - value);
        }
    }
}

// Gradient descent on an image, starting from faint noise. `descent(image, gradient)` fills `gradient` with the
// direction that makes the image worse for the goal, so the image moves against it. Returns nullopt when `cancel` was
// set.
template<typename Descent>
std::optional<Pixels> descend(std::uint32_t seed, int steps, const std::atomic<bool>& cancel, Descent&& descent) {
    std::mt19937 rng{seed};
    std::uniform_real_distribution<float> noise(0.0f, kStartNoise);

    std::vector<float> image(kImageSize);
    for (float& value : image) {
        value = noise(rng);
    }

    std::vector<float> gradient;
    for (int step = 0; step < steps; ++step) {
        if (cancel.load(std::memory_order_relaxed)) {
            return std::nullopt;
        }
        descent(image, gradient);

        // Normalized steps keep moving after the model is already sure and its gradient has gone tiny.
        float largest = 0.0f;
        for (const float g : gradient) {
            largest = std::max(largest, std::abs(g));
        }
        if (largest == 0.0f && step == 0) {
            // Every unit under the neuron is silent for faint noise: start from a brighter image instead.
            std::ranges::fill(image, 0.5f);
            continue;
        }
        const float t = static_cast<float>(step) / static_cast<float>(steps - 1);
        const float rate = kFirstStep + (kLastStep - kFirstStep) * t;
        const float scale = largest > 0.0f ? rate / largest : 0.0f;
        for (std::size_t i = 0; i < kImageSize; ++i) {
            image[i] = std::clamp(image[i] * (1.0f - kDecay) - scale * gradient[i], 0.0f, 1.0f);
        }
        blur(image);
    }

    // Stretch to the full range so a faint result is still readable.
    const float brightest = *std::ranges::max_element(image);
    const float gain = brightest > 0.0f ? 1.0f / brightest : 1.0f;
    Pixels pixels;
    for (std::size_t i = 0; i < kImageSize; ++i) {
        pixels[i] = static_cast<std::uint8_t>(std::lround(std::min(image[i] * gain, 1.0f) * 255.0f));
    }
    return pixels;
}

}

std::optional<Pixels> dream_digit(const Mlp& model, std::size_t digit, const std::atomic<bool>& cancel) {
    Mlp::Workspace workspace = model.make_workspace();
    return descend(static_cast<std::uint32_t>(digit) + 1u, kDigitSteps, cancel,
            [&](const std::vector<float>& image, std::vector<float>& gradient) {
                model.forward_input(image, workspace);
                model.input_gradient(digit, workspace, gradient);
            });
}

std::optional<Pixels> dream_neuron(const Mlp& model, std::size_t layer, std::size_t neuron,
        const std::atomic<bool>& cancel) {
    Mlp::Workspace workspace = model.make_workspace();
    const auto seed = static_cast<std::uint32_t>(1000 + layer * 4096 + neuron);
    return descend(seed, kNeuronSteps, cancel, [&](const std::vector<float>& image, std::vector<float>& gradient) {
        model.forward_input_below(image, layer, workspace);
        model.neuron_input_gradient(layer, neuron, workspace, gradient);
        for (float& g : gradient) {
            g = -g;  // the neuron should fire harder, so the "worse" direction is the opposite one
        }
    });
}

}
