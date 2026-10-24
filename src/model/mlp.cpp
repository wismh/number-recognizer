#include "model/mlp.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <format>
#include <fstream>
#include <random>

namespace game {

namespace {

constexpr char kMagic[4] = {'M', 'N', 'W', 'T'};
// 1: linear model only (weights, then biases). 2: hidden layer sizes in the header; version 1 is version 2 with none.
// 3: the number of epochs trained follows the hidden layer sizes.
constexpr std::uint32_t kVersionLinear = 1;
constexpr std::uint32_t kVersionHidden = 2;
constexpr std::uint32_t kVersionEpochs = 3;

static_assert(std::endian::native == std::endian::little, "model files are little-endian");

template<typename T>
bool read_value(std::istream& in, T& value) {
    return static_cast<bool>(in.read(reinterpret_cast<char*>(&value), sizeof(T)));
}

template<typename T>
void write_value(std::ostream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

}

Mlp::Mlp(std::vector<std::size_t> hidden, NoInit)
    : hidden_(std::move(hidden)) {
    build_layout();
}

Mlp::Mlp(std::vector<std::size_t> hidden)
    : Mlp(std::move(hidden), NoInit{}) {
    std::mt19937 rng{std::random_device{}()};
    for (const Layer& layer : layers_) {
        std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / static_cast<float>(layer.in)));
        for (std::size_t i = 0; i < layer.in * layer.out; ++i) {
            params_[layer.weights + i] = dist(rng);
        }
    }
}

void Mlp::build_layout() {
    std::vector<std::size_t> sizes{kImageSize};
    sizes.insert(sizes.end(), hidden_.begin(), hidden_.end());
    sizes.push_back(kNumClasses);

    layers_.clear();
    std::size_t offset = 0;
    for (std::size_t l = 0; l + 1 < sizes.size(); ++l) {
        Layer layer;
        layer.in = sizes[l];
        layer.out = sizes[l + 1];
        layer.weights = offset;
        layer.biases = offset + layer.in * layer.out;
        offset = layer.biases + layer.out;
        layers_.push_back(layer);
    }
    params_.assign(offset, 0.0f);
}

std::string Mlp::describe() const {
    std::string text = std::to_string(kImageSize);
    for (const std::size_t size : hidden_) {
        text += std::format("-{}", size);
    }
    return text + std::format("-{}", kNumClasses);
}

std::expected<Mlp, std::string> Mlp::load(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return std::unexpected(std::format("cannot open {}", path.string()));
    }

    char magic[4]{};
    in.read(magic, sizeof(magic));
    if (!in || !std::equal(std::begin(magic), std::end(magic), std::begin(kMagic))) {
        return std::unexpected("invalid model file: bad magic");
    }

    std::uint32_t version = 0;
    std::uint32_t image_size = 0;
    std::uint32_t num_classes = 0;
    if (!read_value(in, version) || !read_value(in, image_size) || !read_value(in, num_classes)) {
        return std::unexpected("invalid model file: truncated header");
    }
    if (version < kVersionLinear || version > kVersionEpochs) {
        return std::unexpected(std::format("unsupported model version: {}", version));
    }
    if (image_size != kImageSize || num_classes != kNumClasses) {
        return std::unexpected(std::format("model shape mismatch: expected {}x{}, got {}x{}", kNumClasses,
                kImageSize, num_classes, image_size));
    }

    std::vector<std::size_t> hidden;
    std::uint32_t epochs = 0;
    if (version >= kVersionHidden) {
        std::uint32_t count = 0;
        if (!read_value(in, count) || count > kMaxHiddenLayers) {
            return std::unexpected("invalid model file: bad hidden layer count");
        }
        for (std::uint32_t i = 0; i < count; ++i) {
            std::uint32_t size = 0;
            if (!read_value(in, size) || size == 0 || size > kMaxNeurons) {
                return std::unexpected("invalid model file: bad hidden layer size");
            }
            hidden.push_back(size);
        }
        if (version >= kVersionEpochs && !read_value(in, epochs)) {
            return std::unexpected("invalid model file: truncated header");
        }
    }

    Mlp model{std::move(hidden), NoInit{}};
    model.epochs_trained_ = epochs;
    in.read(reinterpret_cast<char*>(model.params_.data()),
            static_cast<std::streamsize>(model.params_.size() * sizeof(float)));
    if (!in) {
        return std::unexpected("invalid model file: truncated weights");
    }
    return model;
}

std::expected<void, std::string> Mlp::save(const std::filesystem::path& path) const {
    std::error_code error;
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path(), error);
    }

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return std::unexpected(std::format("cannot write {}", path.string()));
    }
    out.write(kMagic, sizeof(kMagic));
    write_value(out, kVersionEpochs);
    write_value(out, static_cast<std::uint32_t>(kImageSize));
    write_value(out, static_cast<std::uint32_t>(kNumClasses));
    write_value(out, static_cast<std::uint32_t>(hidden_.size()));
    for (const std::size_t size : hidden_) {
        write_value(out, static_cast<std::uint32_t>(size));
    }
    write_value(out, static_cast<std::uint32_t>(epochs_trained_));
    out.write(reinterpret_cast<const char*>(params_.data()),
            static_cast<std::streamsize>(params_.size() * sizeof(float)));
    if (!out) {
        return std::unexpected(std::format("write failed: {}", path.string()));
    }
    return {};
}

Mlp::Workspace Mlp::make_workspace() const {
    Workspace workspace;
    workspace.activations.resize(layers_.size() + 1);
    workspace.activations[0].resize(kImageSize);
    for (std::size_t l = 0; l < layers_.size(); ++l) {
        workspace.activations[l + 1].resize(layers_[l].out);
    }
    return workspace;
}

void Mlp::forward(std::span<const std::uint8_t> image, Workspace& workspace) const {
    std::vector<float>& input = workspace.activations[0];
    for (std::size_t i = 0; i < kImageSize; ++i) {
        input[i] = static_cast<float>(image[i]) / 255.0f;
    }
    propagate(workspace);
}

void Mlp::forward_input(std::span<const float> input, Workspace& workspace) const {
    std::ranges::copy(input, workspace.activations[0].begin());
    propagate(workspace);
}

void Mlp::input_gradient(std::size_t label, Workspace& workspace, std::vector<float>& gradient) const {
    workspace.delta = workspace.activations.back();
    workspace.delta[label] -= 1.0f;
    backpropagate_input(layers_.size() - 1, workspace, gradient);
}

std::span<const float> Mlp::incoming_weights(std::size_t layer, std::size_t neuron) const {
    const Layer& l = layers_[layer];
    return {params_.data() + l.weights + neuron * l.in, l.in};
}

float Mlp::bias(std::size_t layer, std::size_t neuron) const {
    return params_[layers_[layer].biases + neuron];
}

void Mlp::forward_input_below(std::span<const float> input, std::size_t layer, Workspace& workspace) const {
    std::ranges::copy(input, workspace.activations[0].begin());
    propagate_layers(workspace, layer);
}

void Mlp::neuron_input_gradient(std::size_t layer, std::size_t neuron, Workspace& workspace,
        std::vector<float>& gradient) const {
    workspace.delta.assign(layers_[layer].out, 0.0f);
    workspace.delta[neuron] = 1.0f;
    backpropagate_input(layer, workspace, gradient);
}

Probabilities Mlp::neuron_influence(std::size_t layer, std::size_t neuron,
        const std::vector<std::vector<float>>* fired) const {
    const std::size_t output = layers_.size() - 1;
    Probabilities influence{};
    std::vector<float> gradient;
    std::vector<float> next;
    for (std::size_t digit = 0; digit < kNumClasses; ++digit) {
        // The score's gradient with respect to the output of the last hidden layer, walked down to `layer`.
        const std::span<const float> row = incoming_weights(output, digit);
        gradient.assign(row.begin(), row.end());
        for (std::size_t m = output - 1; m > layer; --m) {
            const Layer& l = layers_[m];
            next.assign(l.in, 0.0f);
            for (std::size_t o = 0; o < l.out; ++o) {
                if (fired != nullptr && (*fired)[m][o] <= 0.0f) {
                    continue;
                }
                const float* weights = params_.data() + l.weights + o * l.in;
                for (std::size_t i = 0; i < l.in; ++i) {
                    next[i] += weights[i] * gradient[o];
                }
            }
            gradient.swap(next);
        }
        influence[digit] = gradient[neuron];
    }
    return influence;
}

void Mlp::backpropagate_input(std::size_t top_layer, Workspace& workspace, std::vector<float>& gradient) const {
    for (std::size_t l = top_layer + 1; l-- > 0;) {
        const Layer& layer = layers_[l];
        std::vector<float>& next = workspace.next_delta;
        next.assign(layer.in, 0.0f);
        for (std::size_t o = 0; o < layer.out; ++o) {
            const float d = workspace.delta[o];
            if (d == 0.0f) {
                continue;  // dead ReLUs and the one-hot start of a neuron's gradient skip whole rows
            }
            const float* row = params_.data() + layer.weights + o * layer.in;
            for (std::size_t i = 0; i < layer.in; ++i) {
                next[i] += row[i] * d;
            }
        }
        if (l > 0) {
            const std::vector<float>& a = workspace.activations[l];
            for (std::size_t i = 0; i < layer.in; ++i) {
                if (a[i] <= 0.0f) {
                    next[i] = 0.0f;
                }
            }
        }
        workspace.delta.swap(next);
    }
    gradient = workspace.delta;
}

void Mlp::propagate_layers(Workspace& workspace, std::size_t count) const {
    for (std::size_t l = 0; l < count; ++l) {
        const Layer& layer = layers_[l];
        const std::vector<float>& a = workspace.activations[l];
        std::vector<float>& z = workspace.activations[l + 1];
        const bool last = l + 1 == layers_.size();
        for (std::size_t o = 0; o < layer.out; ++o) {
            const float* row = params_.data() + layer.weights + o * layer.in;
            float sum = params_[layer.biases + o];
            for (std::size_t i = 0; i < layer.in; ++i) {
                sum += row[i] * a[i];
            }
            z[o] = last ? sum : std::max(sum, 0.0f);
        }
    }
}

void Mlp::propagate(Workspace& workspace) const {
    propagate_layers(workspace, layers_.size());

    std::vector<float>& out = workspace.activations.back();
    const float max = *std::ranges::max_element(out);
    float total = 0.0f;
    for (float& value : out) {
        value = std::exp(value - max);
        total += value;
    }
    for (float& value : out) {
        value /= total;
    }
}

Probabilities Mlp::probabilities(std::span<const std::uint8_t> image) const {
    Workspace workspace = make_workspace();
    forward(image, workspace);
    Probabilities result{};
    std::ranges::copy(workspace.activations.back(), result.begin());
    return result;
}

std::size_t Mlp::predict(std::span<const std::uint8_t> image) const {
    const Probabilities probs = probabilities(image);
    return static_cast<std::size_t>(std::ranges::max_element(probs) - probs.begin());
}

float Mlp::accumulate_gradient(std::span<const std::uint8_t> image, std::size_t label, std::vector<float>& gradient,
        Workspace& workspace) const {
    forward(image, workspace);
    const float sample_loss = -std::log(std::max(workspace.activations.back()[label], 1e-12f));

    workspace.delta = workspace.activations.back();
    workspace.delta[label] -= 1.0f;

    for (std::size_t l = layers_.size(); l-- > 0;) {
        const Layer& layer = layers_[l];
        const std::vector<float>& a = workspace.activations[l];
        const std::vector<float>& delta = workspace.delta;

        for (std::size_t o = 0; o < layer.out; ++o) {
            float* grad_row = gradient.data() + layer.weights + o * layer.in;
            const float d = delta[o];
            gradient[layer.biases + o] += d;
            for (std::size_t i = 0; i < layer.in; ++i) {
                grad_row[i] += d * a[i];
            }
        }

        if (l == 0) {
            break;
        }
        std::vector<float>& next = workspace.next_delta;
        next.assign(layer.in, 0.0f);
        for (std::size_t o = 0; o < layer.out; ++o) {
            const float* row = params_.data() + layer.weights + o * layer.in;
            const float d = delta[o];
            for (std::size_t i = 0; i < layer.in; ++i) {
                next[i] += row[i] * d;
            }
        }
        for (std::size_t i = 0; i < layer.in; ++i) {
            if (a[i] <= 0.0f) {
                next[i] = 0.0f;
            }
        }
        workspace.delta.swap(next);
    }
    return sample_loss;
}

void Mlp::apply_gradient(const std::vector<float>& gradient, float scale) {
    for (std::size_t i = 0; i < params_.size(); ++i) {
        params_[i] -= scale * gradient[i];
    }
}

}
