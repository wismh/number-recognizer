#include "recognizer.h"

#include <engine/log.h>

#include <algorithm>
#include <chrono>
#include <format>

namespace game {

namespace {

// How long one frame may spend scoring before it hands control back to the UI.
constexpr std::chrono::milliseconds kScoringSlice{8};

std::string percent(float fraction) {
    return std::format("{:.2f}%", fraction * 100.0f);
}

template<typename T>
bool is_ready(const std::future<T>& future) {
    return future.valid() && future.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
}

}

Recognizer::Recognizer(RecognizerViewModel& view, const Tr& tr, ModelStore models, DatasetStore datasets,
        std::filesystem::path data_directory, Parallel parallel)
    : view_(view)
    , tr_(tr)
    , data_directory_(std::move(data_directory))
    , parallel_(std::move(parallel))
    , status_(view, tr)
    , texts_(view, tr)
    , datasets_(view, std::move(datasets), tr, status_)
    , neuron_panel_(view, tr)
    , settings_(view)
    , library_(view, std::move(models))
    , dreamer_(view.prototypePaint)
    , session_(status_, parallel_) {
    view_.train.bind_to<Recognizer, &Recognizer::train, &Recognizer::can_train>(*this);
    view_.continueTraining.bind_to<Recognizer, &Recognizer::continue_training, &Recognizer::can_continue>(*this);
    view_.loadModel.bind_to<Recognizer, &Recognizer::load_model, &Recognizer::can_load>(*this);
    view_.deleteModel.bind_to<Recognizer, &Recognizer::delete_model, &Recognizer::can_load>(*this);
    view_.refreshModels.bind_to<Recognizer, &Recognizer::refresh_models, &Recognizer::can_refresh>(*this);
    view_.clearCanvas.bind_to<Recognizer, &Recognizer::clear_canvas, &Recognizer::can_clear>(*this);
    view_.addSample.bind_to<Recognizer, &Recognizer::add_sample, &Recognizer::can_add>(*this);

    view_.probabilityPaint.reset();
    texts_.render();
    datasets_.apply_locale();

    dataset_load_ = std::async(std::launch::async, [directory = data_directory_] { return MnistData::load(directory); });
    status_.say("status.loading_mnist");
}

void Recognizer::apply_locale() {
    texts_.render();
    datasets_.apply_locale();
    neuron_panel_.apply_locale();
    status_.refresh();
}

void Recognizer::update() {
    dreamer_.collect();
    neuron_panel_.update(view_.tabNeurons.get());
    if (is_ready(dataset_load_)) {
        finish_dataset_load();
    }
    if (auto loaded = library_.poll_load()) {
        finish_model_load(std::move(*loaded));
    }
    if (is_ready(save_)) {
        finish_save();
    }
    if (auto result = session_.update()) {
        finish_training(std::move(*result));
    }
    if (evaluation_) {
        if (evaluation_->step(kScoringSlice)) {
            finish_evaluation();
        } else {
            texts_.scoring(evaluation_->progress());
        }
    }
}

void Recognizer::draw_at(float nx, float ny) {
    view_.canvasPaint.stamp(nx, ny);
}

void Recognizer::canvas_changed() {
    if (model_) {
        run_prediction();
    }
}

bool Recognizer::busy() const {
    return session_.active() || library_.loading() || save_.valid() || evaluation_.has_value();
}

bool Recognizer::has_training_samples() const {
    const auto samples = datasets_.active_samples();
    return samples && samples->size() > 0;
}

bool Recognizer::can_train() const {
    return mnist_ && has_training_samples() && !busy();
}

bool Recognizer::can_continue() const {
    return model_ && mnist_ && has_training_samples() && !busy();
}

bool Recognizer::can_load() const {
    return library_.has_selection() && !busy();
}

// MNIST test is the one fixed yardstick, whatever a model was trained on. The pointer shares MnistData's ownership.
std::shared_ptr<const LabeledImages> Recognizer::mnist_test() const {
    return std::shared_ptr<const LabeledImages>(mnist_, &mnist_->test);
}

void Recognizer::train() {
    Mlp model{settings_.hidden_layers()};
    std::string label = model.label();
    start_training(std::move(model), label);
}

// Trains a copy of the model in memory; the result is saved as a new file next to the original.
void Recognizer::continue_training() {
    start_training(Mlp{*model_}, model_->label() + "_ft");
}

void Recognizer::start_training(Mlp model, const std::string& label) {
    session_.start(std::move(model), label, datasets_.active_name(), datasets_.active_samples(), settings_.epochs(),
            settings_.augmentation());
    settings_.set_locked(true);
    texts_.set_training(true);
}

void Recognizer::finish_training(TrainingSession::Result result) {
    settings_.set_locked(false);
    texts_.set_training(false);

    RecognizerTexts::LastTraining last;
    last.dataset = std::move(result.dataset);
    last.samples = result.samples->size();
    last.loss = result.loss;
    last.file = std::move(result.file);

    // The training set is scored once, now; a model loaded later is scored on MNIST test only.
    texts_.set_last_training(last);
    set_loaded_model(std::make_shared<const Mlp>(std::move(result.model)), last.file, std::move(result.samples));

    // The file is written on another thread and the model scored in slices; report once both are done.
    reporting_ = true;
    if (library_.can_save()) {
        save_ = library_.save(model_, last.file);
    } else {
        last.save_error = tr_("err.no_user_data");
        texts_.set_last_training(last);
    }
    status_.say("status.saving");
}

void Recognizer::set_loaded_model(std::shared_ptr<const Mlp> model, const std::string& name,
        std::shared_ptr<const LabeledImages> train_set) {
    model_ = std::move(model);
    view_.modelLoaded = true;
    view_.modelMissing = false;
    texts_.set_model({.name = name,
            .architecture = model_->describe(),
            .hidden = model_->hidden(),
            .epochs = model_->epochs_trained()});
    neuron_panel_.set_model(model_);
    run_prediction();
    start_evaluation(std::move(train_set));
}

void Recognizer::start_evaluation(std::shared_ptr<const LabeledImages> train_set) {
    dreamer_.start(model_);
    texts_.waiting_for_scores(mnist_ != nullptr);
    if (!mnist_) {
        return;
    }
    std::vector<std::shared_ptr<const LabeledImages>> sets{mnist_test()};
    evaluating_train_set_ = train_set != nullptr;
    if (train_set) {
        sets.push_back(std::move(train_set));
    }
    evaluation_.emplace(model_, std::move(sets), parallel_);
    texts_.scoring(0.0f);
}

void Recognizer::finish_evaluation() {
    RecognizerTexts::Scores scores;
    scores.test = evaluation_->accuracy(0);
    scores.mutated = evaluation_->mutated_accuracy(0);
    for (std::size_t digit = 0; digit < kNumClasses; ++digit) {
        scores.total[digit] = evaluation_->digit_total(0, digit);
        scores.correct[digit] = evaluation_->digit_correct(0, digit);
    }
    test_accuracy_ = scores.test;
    texts_.set_scores(scores);

    if (evaluating_train_set_ && texts_.last_training() != nullptr) {
        RecognizerTexts::LastTraining last = *texts_.last_training();
        last.scored = true;
        last.train = evaluation_->accuracy(1);
        last.train_mutated = evaluation_->mutated_accuracy(1);
        texts_.set_last_training(std::move(last));
    }
    evaluation_.reset();
    report_training();
}

void Recognizer::finish_save() {
    if (const auto saved = save_.get(); !saved && texts_.last_training() != nullptr) {
        RecognizerTexts::LastTraining last = *texts_.last_training();
        last.save_error = saved.error();
        texts_.set_last_training(std::move(last));
    }
    report_training();
}

// Called whenever the save or the scoring finishes; speaks once both have.
void Recognizer::report_training() {
    if (!reporting_ || save_.valid() || evaluation_ || texts_.last_training() == nullptr) {
        return;
    }
    reporting_ = false;
    const RecognizerTexts::LastTraining& last = *texts_.last_training();
    if (last.save_error.empty()) {
        library_.rebuild(last.file);
        status_.say("status.saved", {{"file", last.file},
                                            {"loss", std::format("{:.4f}", last.loss)},
                                            {"test", percent(test_accuracy_)}});
    } else {
        status_.say("status.save_failed", {{"detail", last.save_error}});
    }
}

void Recognizer::finish_dataset_load() {
    auto loaded = dataset_load_.get();
    if (!loaded) {
        engine::log::warn("MNIST: " + loaded.error());
        status_.say("status.mnist_failed", {{"detail", loaded.error()}});
        return;
    }
    mnist_ = std::make_shared<const MnistData>(std::move(*loaded));
    datasets_.set_mnist(std::shared_ptr<const LabeledImages>(mnist_, &mnist_->train));
    status_.say("status.mnist_ready", {{"train", std::to_string(mnist_->train.size())},
                                              {"test", std::to_string(mnist_->test.size())}});
}

void Recognizer::load_model() {
    library_.start_load();
    status_.say("status.model_loading", {{"name", library_.selected_name()}});
}

void Recognizer::finish_model_load(ModelLibrary::Loaded loaded) {
    if (!loaded.model) {
        status_.say("status.model_load_failed", {{"detail", loaded.model.error()}});
        return;
    }
    Mlp model = std::move(*loaded.model);
    if (model.epochs_trained() == 0) {
        model.set_epochs_trained(ModelStore::epochs_in_name(loaded.name));
    }
    set_loaded_model(std::make_shared<const Mlp>(std::move(model)), loaded.name);
    status_.say("status.model_loaded", {{"name", loaded.name}, {"arch", model_->describe()}});
}

// The model in memory keeps predicting after its file is deleted, until another is loaded.
void Recognizer::delete_model() {
    const std::string name = library_.selected_name();
    if (!library_.remove_selected()) {
        status_.say("status.model_delete_failed", {{"name", name}});
        return;
    }
    status_.say("status.model_deleted", {{"name", name}});
}

void Recognizer::refresh_models() {
    library_.refresh();
    status_.say("status.models_refreshed");
}

void Recognizer::clear_canvas() {
    view_.canvasPaint.clear();
    texts_.set_prediction(std::nullopt);
    view_.probabilityPaint.reset();
    if (model_) {
        neuron_panel_.clear_drawing();
    }
}

void Recognizer::add_sample() {
    if (datasets_.add_canvas_digit()) {
        clear_canvas();
    }
}

void Recognizer::run_prediction() {
    const Pixels& pixels = view_.canvasPaint.pixels();
    Mlp::Workspace workspace = model_->make_workspace();
    model_->forward(pixels, workspace);
    Probabilities probs{};
    std::ranges::copy(workspace.activations.back(), probs.begin());
    view_.probabilityPaint.set(probs);
    texts_.set_prediction(static_cast<std::size_t>(std::ranges::max_element(probs) - probs.begin()));

    if (is_blank(pixels)) {
        neuron_panel_.clear_drawing();
    } else {
        neuron_panel_.set_drawing(pixels, workspace);
    }
}

}
