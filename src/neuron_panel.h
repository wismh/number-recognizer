#pragma once

#include "loc/tr.h"
#include "model/mlp.h"
#include "model/neuron_dreamer.h"
#include "view/recognizer_view_model.h"

#include <memory>
#include <optional>
#include <vector>

namespace game {

// The Neurons tab: what each neuron of the loaded model looks for. One layer at a time is a grid of pictures, one per
// neuron, in one of two kinds: the neuron's weights drawn on the 28x28 pixels it reads (only for neurons that read
// pixels), or its dream, the image it fires at hardest (for any neuron, found in the background). Picking a neuron shows
// it large, with a chart of how it pushes each digit up or down. The state is kept as numbers so that `apply_locale` can write the texts
// again in another language.
class NeuronPanel {
public:
    NeuronPanel(RecognizerViewModel& view, const Tr& tr);

    NeuronPanel(const NeuronPanel&) = delete;
    NeuronPanel& operator=(const NeuronPanel&) = delete;

    // A different model is loaded: back to its first layer.
    void set_model(std::shared_ptr<const Mlp> model);
    // The model has just run on the drawing; `workspace` holds what every neuron answered.
    void set_drawing(const Pixels& pixels, const Mlp::Workspace& workspace);
    void clear_drawing();
    // Once per frame on the main thread: takes in finished dreams and, while the tab is on screen, updates what depends
    // on the drawing.
    void update(bool visible);
    void apply_locale();

private:
    enum class Mode { Weights, Dream };

    [[nodiscard]] bool can_layer_up() const { return model_ && layer_ + 1 < layer_count(); }
    [[nodiscard]] bool can_layer_down() const { return model_ && layer_ > 0; }
    [[nodiscard]] bool can_show_weights() const { return model_ && weights_available(); }
    [[nodiscard]] bool can_show_dream() const { return model_ != nullptr; }

    // A model without hidden layers has just the output layer to show: one neuron per digit.
    [[nodiscard]] std::size_t layer_count() const;
    [[nodiscard]] bool is_output_layer() const { return layer_ + 1 == model_->layer_count(); }
    // Only neurons that read the pixels have weights that can be drawn as a picture.
    [[nodiscard]] bool weights_available() const { return model_->layer_inputs(layer_) == kImageSize; }
    [[nodiscard]] PicturePaint& cell(std::size_t neuron);

    void layer_up();
    void layer_down();
    void show_weights() { set_mode(Mode::Weights); }
    void show_dream() { set_mode(Mode::Dream); }
    void set_mode(Mode mode);
    void enter_layer(std::size_t layer);
    void select(std::size_t neuron);

    void build_grid();
    void show_cell(std::size_t neuron);
    void show_detail();
    void start_dreaming();
    void harvest();
    void refresh_meters();

    void render();
    void render_layer_text();
    void render_progress();
    void render_detail_text();

    RecognizerViewModel& view_;
    const Tr& tr_;

    std::shared_ptr<const Mlp> model_;
    std::size_t layer_ = 0;
    std::size_t selected_ = 0;
    Mode mode_ = Mode::Weights;
    Mode preferred_ = Mode::Weights;
    // Of the current layer, when its neurons read pixels.
    std::vector<SignedPixels> weights_;
    // Per layer, per neuron: the dreams worked out so far, and how many of them there are.
    std::vector<std::vector<std::optional<Pixels>>> dreams_;
    std::vector<std::size_t> dream_counts_;
    std::size_t dream_layer_ = 0;
    NeuronDreamer dreamer_;
    std::vector<std::shared_ptr<NeuronRow>> rows_;

    // What every layer answered to the drawing on the canvas.
    bool has_drawing_ = false;
    std::vector<std::vector<float>> drawing_activations_;
    bool dirty_ = false;
};

}
