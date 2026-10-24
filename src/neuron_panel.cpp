#include "neuron_panel.h"

#include "model/weight_picture.h"

#include <algorithm>
#include <array>
#include <format>

namespace game {

NeuronPanel::NeuronPanel(RecognizerViewModel& view, const Tr& tr)
    : view_(view)
    , tr_(tr) {
    view_.neuronsLayerUp.bind_to<NeuronPanel, &NeuronPanel::layer_up, &NeuronPanel::can_layer_up>(*this);
    view_.neuronsLayerDown.bind_to<NeuronPanel, &NeuronPanel::layer_down, &NeuronPanel::can_layer_down>(*this);
    view_.showWeights.bind_to<NeuronPanel, &NeuronPanel::show_weights, &NeuronPanel::can_show_weights>(*this);
    view_.showDream.bind_to<NeuronPanel, &NeuronPanel::show_dream, &NeuronPanel::can_show_dream>(*this);
}

std::size_t NeuronPanel::layer_count() const {
    return model_->hidden().empty() ? 1 : model_->hidden().size();
}

PicturePaint& NeuronPanel::cell(std::size_t neuron) {
    return rows_[neuron / NeuronRow::kCells]->cell(neuron % NeuronRow::kCells);
}

void NeuronPanel::set_model(std::shared_ptr<const Mlp> model) {
    dreamer_.stop();
    model_ = std::move(model);
    has_drawing_ = false;
    drawing_activations_.clear();

    dreams_.assign(layer_count(), {});
    dream_counts_.assign(layer_count(), 0);
    for (std::size_t layer = 0; layer < layer_count(); ++layer) {
        dreams_[layer].resize(model_->layer_size(layer));
    }
    selected_ = 0;
    preferred_ = Mode::Weights;
    enter_layer(0);
}

void NeuronPanel::set_drawing(const Pixels& pixels, const Mlp::Workspace& workspace) {
    view_.drawingPaint.show_gray(pixels);
    drawing_activations_.resize(layer_count());
    for (std::size_t layer = 0; layer < layer_count(); ++layer) {
        drawing_activations_[layer] = workspace.activations[layer + 1];
    }
    has_drawing_ = true;
    dirty_ = true;
}

void NeuronPanel::clear_drawing() {
    view_.drawingPaint.show_gray(Pixels{});
    has_drawing_ = false;
    dirty_ = true;
}

void NeuronPanel::update(bool visible) {
    if (!model_) {
        return;
    }
    harvest();
    if (visible && dirty_) {
        dirty_ = false;
        refresh_meters();
        render_detail_text();
    }
}

void NeuronPanel::apply_locale() {
    if (model_) {
        render();
    }
}

void NeuronPanel::layer_up() {
    enter_layer(layer_ + 1);
}

void NeuronPanel::layer_down() {
    enter_layer(layer_ - 1);
}

void NeuronPanel::set_mode(Mode mode) {
    if (mode == mode_) {
        return;
    }
    harvest();
    mode_ = mode;
    preferred_ = mode;
    if (mode_ == Mode::Dream) {
        start_dreaming();
    } else {
        dreamer_.stop();
    }
    for (std::size_t neuron = 0; neuron < model_->layer_size(layer_); ++neuron) {
        show_cell(neuron);
    }
    show_detail();
    render();
}

// Switches to `layer`: its pictures, its grid, and the dreams it still lacks.
void NeuronPanel::enter_layer(std::size_t layer) {
    if (!rows_.empty()) {
        harvest();
    }
    dreamer_.stop();
    layer_ = layer;
    selected_ = std::min(selected_, model_->layer_size(layer_) - 1);
    // Deeper layers only have dreams; the player's choice comes back on a layer that has both.
    mode_ = weights_available() ? preferred_ : Mode::Dream;

    weights_.clear();
    if (weights_available()) {
        for (std::size_t neuron = 0; neuron < model_->layer_size(layer_); ++neuron) {
            weights_.push_back(weight_picture(*model_, layer_, neuron));
        }
    }
    build_grid();
    if (mode_ == Mode::Dream) {
        start_dreaming();
    }
    for (std::size_t neuron = 0; neuron < model_->layer_size(layer_); ++neuron) {
        show_cell(neuron);
    }
    cell(selected_).set_selected(true);
    dirty_ = true;
    show_detail();
    refresh_meters();
    render();
}

void NeuronPanel::select(std::size_t neuron) {
    cell(selected_).set_selected(false);
    selected_ = neuron;
    cell(selected_).set_selected(true);
    if (mode_ == Mode::Dream && !dreams_[layer_][neuron]) {
        dreamer_.prioritize(neuron);
    }
    show_detail();
}

void NeuronPanel::build_grid() {
    const std::size_t count = model_->layer_size(layer_);
    const bool digits = is_output_layer();
    rows_.clear();
    for (std::size_t first = 0; first < count; first += NeuronRow::kCells) {
        auto row = std::make_shared<NeuronRow>(std::min(NeuronRow::kCells, count - first));
        for (std::size_t column = 0; column < NeuronRow::kCells && first + column < count; ++column) {
            const std::size_t neuron = first + column;
            row->pick(column) = [this, neuron] { select(neuron); };
            row->cell(column).set_label(static_cast<int>(digits ? neuron : neuron + 1));
        }
        rows_.push_back(std::move(row));
    }
    view_.neuronRows.set(rows_);
}

void NeuronPanel::show_cell(std::size_t neuron) {
    if (mode_ == Mode::Weights) {
        cell(neuron).show(weights_[neuron], PicturePaint::Palette::Diverging);
    } else if (const auto& dream = dreams_[layer_][neuron]) {
        cell(neuron).show_gray(*dream);
    } else {
        cell(neuron).hide();
    }
}

void NeuronPanel::show_detail() {
    if (mode_ == Mode::Weights) {
        view_.neuronPaint.show(weights_[selected_], PicturePaint::Palette::Diverging);
    } else if (const auto& dream = dreams_[layer_][selected_]) {
        view_.neuronPaint.show_gray(*dream);
    } else {
        view_.neuronPaint.hide();
    }
    render_detail_text();
}

// Works on the neurons of the current layer that have no dream yet, the picked one first.
void NeuronPanel::start_dreaming() {
    dream_layer_ = layer_;
    std::vector<std::size_t> missing;
    if (!dreams_[layer_][selected_]) {
        missing.push_back(selected_);
    }
    for (std::size_t neuron = 0; neuron < model_->layer_size(layer_); ++neuron) {
        if (neuron != selected_ && !dreams_[layer_][neuron]) {
            missing.push_back(neuron);
        }
    }
    dreamer_.start(model_, layer_, std::move(missing));
}

void NeuronPanel::harvest() {
    const auto pictures = dreamer_.collect();
    for (const NeuronDreamer::Picture& picture : pictures) {
        auto& slot = dreams_[dream_layer_][picture.neuron];
        if (!slot) {
            ++dream_counts_[dream_layer_];
        }
        slot = picture.pixels;
        if (mode_ == Mode::Dream && layer_ == dream_layer_) {
            cell(picture.neuron).show_gray(picture.pixels);
            if (picture.neuron == selected_) {
                view_.neuronPaint.show_gray(picture.pixels);
            }
        }
    }
    if (!pictures.empty()) {
        render_progress();
    }
}

// The bar under each picture: how strongly the neuron reacts to the drawing, next to the strongest of its layer.
void NeuronPanel::refresh_meters() {
    const std::size_t count = model_->layer_size(layer_);
    if (!has_drawing_) {
        for (std::size_t neuron = 0; neuron < count; ++neuron) {
            cell(neuron).set_meter(-1.0f);
        }
        return;
    }
    const std::vector<float>& values = drawing_activations_[layer_];
    // The output layer holds probabilities already.
    const float strongest = is_output_layer() ? 1.0f : *std::ranges::max_element(values);
    for (std::size_t neuron = 0; neuron < count; ++neuron) {
        cell(neuron).set_meter(strongest > 0.0f ? values[neuron] / strongest : 0.0f);
    }
}

void NeuronPanel::render() {
    render_layer_text();
    render_progress();
    render_detail_text();
}

void NeuronPanel::render_layer_text() {
    const std::string size = std::to_string(model_->layer_size(layer_));
    view_.neuronsLayerText = is_output_layer()
            ? tr_("neurons.layer_output", {{"size", size}})
            : tr_("neurons.layer", {{"layer", std::to_string(layer_ + 1)}, {"count", std::to_string(layer_count())},
                      {"size", size}});
}

// The mode buttons and the line beside them: how far the dreams are, or why Weights is not offered.
void NeuronPanel::render_progress() {
    view_.weightsActive = mode_ == Mode::Weights;
    view_.weightsIdle = mode_ != Mode::Weights;
    view_.dreamActive = mode_ == Mode::Dream;
    view_.dreamIdle = mode_ != Mode::Dream;

    const std::size_t total = model_->layer_size(layer_);
    if (mode_ == Mode::Dream && dream_counts_[layer_] < total) {
        view_.neuronsProgressText = tr_("neurons.progress",
                {{"done", std::to_string(dream_counts_[layer_])}, {"total", std::to_string(total)}});
    } else if (!weights_available()) {
        view_.neuronsProgressText = tr_("neurons.note_deep");
    } else {
        view_.neuronsProgressText = std::string{};
    }
}

void NeuronPanel::render_detail_text() {
    view_.neuronTitleText = is_output_layer()
            ? tr_("neurons.title_digit", {{"digit", std::to_string(selected_)}})
            : tr_("neurons.title",
                      {{"layer", std::to_string(layer_ + 1)}, {"neuron", std::to_string(selected_ + 1)}});
    view_.neuronLegendText = tr_(mode_ == Mode::Weights ? "neurons.legend_weights" : "neurons.legend_dream");
    view_.neuronBiasText = tr_("neurons.bias", {{"value", std::format("{:+.2f}", model_->bias(layer_, selected_))}});

    if (!has_drawing_) {
        view_.neuronNowText = tr_("neurons.no_drawing");
    } else {
        const std::vector<float>& values = drawing_activations_[layer_];
        const float value = values[selected_];
        if (is_output_layer()) {
            view_.neuronNowText = tr_("neurons.now_prob", {{"percent", std::format("{:.0f}", value * 100.0f)}});
        } else if (value <= 0.0f) {
            view_.neuronNowText = tr_("neurons.silent");
        } else {
            const float share = value / *std::ranges::max_element(values);
            view_.neuronNowText = tr_("neurons.now",
                    {{"value", std::format("{:.2f}", value)}, {"percent", std::format("{:.0f}", share * 100.0f)}});
        }
    }

    // A neuron of the output layer is a digit itself; any other one pushes the digits up or down. The last hidden layer
    // is wired straight to them. Deeper ones act through the layers above, which follow what fired for the drawing.
    view_.hasInfluence = !is_output_layer();
    if (!is_output_layer()) {
        const bool direct = layer_ + 2 == model_->layer_count();
        view_.influencePaint.set(
                model_->neuron_influence(layer_, selected_, has_drawing_ ? &drawing_activations_ : nullptr));
        view_.neuronInfluenceText = tr_(direct ? "neurons.influence"
                        : has_drawing_         ? "neurons.influence_drawing"
                                               : "neurons.influence_open");
    }
}

}
