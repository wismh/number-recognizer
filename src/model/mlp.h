#pragma once

#include "model/digits.h"

#include <expected>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace game {

// Fully connected network: 784 inputs (pixels / 255), `hidden` ReLU layers, 10 softmax outputs. With no hidden layer
// it is a linear softmax classifier. All weights and biases live in one flat vector, layer after layer, each layer as
// `out` rows of `in` weights followed by `out` biases.
class Mlp {
public:
    static constexpr std::size_t kMaxHiddenLayers = 8;
    static constexpr std::size_t kMaxNeurons = 1024;

    // Per-thread scratch for forward and backward passes.
    struct Workspace {
        std::vector<std::vector<float>> activations;  // input, every hidden layer, output
        std::vector<float> delta;
        std::vector<float> next_delta;
    };

    // He-initialized weights, zero biases.
    explicit Mlp(std::vector<std::size_t> hidden);

    [[nodiscard]] static std::expected<Mlp, std::string> load(const std::filesystem::path& path);
    [[nodiscard]] std::expected<void, std::string> save(const std::filesystem::path& path) const;

    [[nodiscard]] const std::vector<std::size_t>& hidden() const { return hidden_; }
    // "784-64-64-10".
    [[nodiscard]] std::string describe() const;
    [[nodiscard]] std::size_t parameter_count() const { return params_.size(); }
    // Epochs of training this model has had in total, across Train and Continue; 0 when unknown (older files).
    [[nodiscard]] unsigned epochs_trained() const { return epochs_trained_; }
    void set_epochs_trained(unsigned epochs) { epochs_trained_ = epochs; }

    [[nodiscard]] Workspace make_workspace() const;
    // Fills `workspace.activations`; the last entry holds the probabilities.
    void forward(std::span<const std::uint8_t> image, Workspace& workspace) const;
    // Same for an input already scaled to 0..1 (kImageSize values).
    void forward_input(std::span<const float> input, Workspace& workspace) const;
    // After a forward pass: the gradient of -log(p[label]) with respect to the input values, so that stepping the
    // input against it makes the model more sure of `label`.
    void input_gradient(std::size_t label, Workspace& workspace, std::vector<float>& gradient) const;

    // Layers counted from the first: the hidden layers, then the output layer.
    [[nodiscard]] std::size_t layer_count() const { return layers_.size(); }
    [[nodiscard]] std::size_t layer_size(std::size_t layer) const { return layers_[layer].out; }
    [[nodiscard]] std::size_t layer_inputs(std::size_t layer) const { return layers_[layer].in; }
    // What `neuron` of `layer` multiplies each of the layer's inputs by, and what it adds after that.
    [[nodiscard]] std::span<const float> incoming_weights(std::size_t layer, std::size_t neuron) const;
    [[nodiscard]] float bias(std::size_t layer, std::size_t neuron) const;
    // Same as forward_input, but stops before `layer`: only the layers under it are worked out.
    void forward_input_below(std::span<const float> input, std::size_t layer, Workspace& workspace) const;
    // After forward_input_below(.., layer, ..): the gradient of the neuron's weighted sum (before ReLU) with respect
    // to the input values, so that stepping the input along it makes the neuron fire harder.
    void neuron_input_gradient(std::size_t layer, std::size_t neuron, Workspace& workspace,
            std::vector<float>& gradient) const;

    // How much each digit's score moves per unit of activation of `neuron` in hidden `layer`, through the layers above
    // it. `fired` is what every hidden layer answered to some input: a unit that stayed at 0 passes nothing on. Null
    // counts every unit as firing. For the last hidden layer this is just the neuron's weights to the digits.
    [[nodiscard]] Probabilities neuron_influence(std::size_t layer, std::size_t neuron,
            const std::vector<std::vector<float>>* fired) const;

    [[nodiscard]] Probabilities probabilities(std::span<const std::uint8_t> image) const;
    [[nodiscard]] std::size_t predict(std::span<const std::uint8_t> image) const;
    // Adds the cross-entropy gradient of one sample into `gradient` (parameter_count() floats). Returns the
    // sample's loss.
    float accumulate_gradient(std::span<const std::uint8_t> image, std::size_t label, std::vector<float>& gradient,
            Workspace& workspace) const;
    // params -= scale * gradient.
    void apply_gradient(const std::vector<float>& gradient, float scale);

private:
    struct Layer {
        std::size_t in = 0;
        std::size_t out = 0;
        std::size_t weights = 0;  // offset of the weights in params_
        std::size_t biases = 0;   // offset of the biases in params_
    };
    struct NoInit {};

    Mlp(std::vector<std::size_t> hidden, NoInit);
    void build_layout();
    // Runs the first `count` layers on workspace.activations[0], ReLU on all but the output layer.
    void propagate_layers(Workspace& workspace, std::size_t count) const;
    // Runs every layer, then softmax.
    void propagate(Workspace& workspace) const;
    // `workspace.delta` holds the gradient with respect to the weighted sums of `top_layer`; carries it down through
    // the layers below to the input.
    void backpropagate_input(std::size_t top_layer, Workspace& workspace, std::vector<float>& gradient) const;

    std::vector<std::size_t> hidden_;
    std::vector<Layer> layers_;
    std::vector<float> params_;
    unsigned epochs_trained_ = 0;
};

}
