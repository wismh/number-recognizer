#include "recognizer.h"

#include <engine/log.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <format>

namespace game {

namespace {

// How long one frame may spend training before it hands control back to the UI.
constexpr std::chrono::milliseconds kTrainingSlice{8};

// "linear", "<layers>x<neurons>" when all hidden layers are equal, else the sizes joined with dashes.
std::string label_of(const Mlp& model) {
    const std::vector<std::size_t>& hidden = model.hidden();
    if (hidden.empty()) {
        return "linear";
    }
    if (std::ranges::all_of(hidden, [&](std::size_t size) { return size == hidden.front(); })) {
        return std::format("{}x{}", hidden.size(), hidden.front());
    }
    std::string label;
    for (const std::size_t size : hidden) {
        label += (label.empty() ? "" : "-") + std::to_string(size);
    }
    return label;
}

std::string percent(float fraction) {
    return std::format("{:.2f}%", fraction * 100.0f);
}

// Files written before the epoch count was stored carry it in their name: model_<arch>_<n>ep_<time>.bin.
unsigned epochs_in_name(const std::string& name) {
    const std::size_t end = name.find("ep_");
    if (end == std::string::npos) {
        return 0;
    }
    std::size_t begin = end;
    while (begin > 0 && std::isdigit(static_cast<unsigned char>(name[begin - 1]))) {
        --begin;
    }
    return begin == end ? 0 : static_cast<unsigned>(std::stoul(name.substr(begin, end - begin)));
}

}

Recognizer::Recognizer(RecognizerViewModel& view, const Tr& tr, ModelStore models, DatasetStore datasets,
        std::filesystem::path data_directory, engine::JobSystem& jobs)
    : view_(view)
    , tr_(tr)
    , store_(std::move(models))
    , data_directory_(std::move(data_directory))
    , parallel_{static_cast<std::size_t>(jobs.worker_count()) + 1,
              [&jobs](std::size_t count, std::size_t grain, const Parallel::RangeFn& fn) {
                  jobs.parallel_for(count, grain, [&fn](std::size_t begin, std::size_t end) { fn(begin, end); });
              }}
    , status_(view, tr)
    , texts_(view, tr)
    , datasets_(view, std::move(datasets), tr, status_)
    , neuron_panel_(view, tr) {
    view_.train.bind_to<Recognizer, &Recognizer::train, &Recognizer::can_train>(*this);
    view_.continueTraining.bind_to<Recognizer, &Recognizer::continue_training, &Recognizer::can_continue>(*this);
    view_.loadModel.bind_to<Recognizer, &Recognizer::load_model, &Recognizer::can_load>(*this);
    view_.deleteModel.bind_to<Recognizer, &Recognizer::delete_model, &Recognizer::can_load>(*this);
    view_.refreshModels.bind_to<Recognizer, &Recognizer::refresh_models, &Recognizer::can_refresh>(*this);
    view_.clearCanvas.bind_to<Recognizer, &Recognizer::clear_canvas, &Recognizer::can_clear>(*this);
    view_.addSample.bind_to<Recognizer, &Recognizer::add_sample, &Recognizer::can_add>(*this);
    view_.epochsUp.bind_to<Recognizer, &Recognizer::epochs_up, &Recognizer::can_epochs_up>(*this);
    view_.epochsDown.bind_to<Recognizer, &Recognizer::epochs_down, &Recognizer::can_epochs_down>(*this);
    view_.layersUp.bind_to<Recognizer, &Recognizer::layers_up, &Recognizer::can_layers_up>(*this);
    view_.layersDown.bind_to<Recognizer, &Recognizer::layers_down, &Recognizer::can_layers_down>(*this);
    view_.neuronsUp.bind_to<Recognizer, &Recognizer::neurons_up, &Recognizer::can_neurons_up>(*this);
    view_.neuronsDown.bind_to<Recognizer, &Recognizer::neurons_down, &Recognizer::can_neurons_down>(*this);

    view_.epochsText = std::to_string(epochs_);
    show_architecture();
    view_.probabilityPaint.reset();
    rebuild_model_list({});
    texts_.render();
    datasets_.apply_locale();

    dataset_load_ = std::async(std::launch::async, [directory = data_directory_] { return MnistData::load(directory); });
    status_.say("status.loading_mnist");
}

Recognizer::~Recognizer() {
    stop_dreaming();
}

void Recognizer::apply_locale() {
    texts_.render();
    datasets_.apply_locale();
    neuron_panel_.apply_locale();
    status_.refresh();
}

void Recognizer::update() {
    collect_dreams();
    neuron_panel_.update(view_.tabNeurons.get());
    if (dataset_load_.valid() && dataset_load_.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        finish_dataset_load();
    }
    if (model_load_.valid() && model_load_.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        finish_model_load();
    }
    if (save_.valid() && save_.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        finish_save();
    }
    if (run_) {
        if (run_->step(kTrainingSlice)) {
            finish_training();
        } else {
            show_training_progress();
        }
    }
    if (evaluation_) {
        if (evaluation_->step(kTrainingSlice)) {
            finish_evaluation();
        } else {
            texts_.scoring(evaluation_->progress());
        }
    }
}

void Recognizer::draw_at(float nx, float ny) {
    view_.canvasPaint.stamp(nx, ny);
    if (model_) {
        run_prediction();
    }
}

void Recognizer::end_stroke() {
    if (model_) {
        run_prediction();
    }
}

bool Recognizer::busy() const {
    return training() || model_load_.valid() || save_.valid() || evaluation_.has_value();
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
    return selected_ && !busy();
}

std::vector<std::size_t> Recognizer::hidden_layers() const {
    return std::vector<std::size_t>(layers_, neurons_);
}

// MNIST test is the one fixed yardstick, whatever a model was trained on. The pointer shares MnistData's ownership.
std::shared_ptr<const LabeledImages> Recognizer::mnist_test() const {
    return std::shared_ptr<const LabeledImages>(mnist_, &mnist_->test);
}

void Recognizer::train() {
    Mlp model{hidden_layers()};
    std::string label = label_of(model);
    start_training(std::move(model), std::move(label));
}

// Trains a copy of the model in memory; the result is saved as a new file next to the original.
void Recognizer::continue_training() {
    start_training(Mlp{*model_}, label_of(*model_) + "_ft");
}

void Recognizer::start_training(Mlp model, const std::string& label) {
    run_samples_ = datasets_.active_samples();
    run_dataset_name_ = datasets_.active_name();
    run_file_name_ = ModelStore::new_file_name(label, model.epochs_trained() + epochs_);
    const Augmentation augmentation{.shift = view_.shiftAugment.get(), .scale = view_.scaleAugment.get()};
    run_.emplace(std::move(model), run_samples_, epochs_, augmentation, parallel_);
    texts_.set_training(true);
    show_training_progress();
}

void Recognizer::show_training_progress() {
    const unsigned done = run_->completed_epochs();
    Tr::Args args{{"dataset", run_dataset_name_},
            {"epoch", std::to_string(std::min(done + 1, run_->epochs()))},
            {"epochs", std::to_string(run_->epochs())},
            {"percent", std::format("{:.0f}", run_->epoch_progress() * 100.0f)}};
    if (done > 0) {
        args.emplace_back("loss", std::format("{:.4f}", run_->last_loss()));
        status_.say("status.training_loss", std::move(args));
    } else {
        status_.say("status.training", std::move(args));
    }
}

void Recognizer::finish_training() {
    Mlp trained = run_->take_model();
    trained.set_epochs_trained(trained.epochs_trained() + run_->epochs());
    RecognizerTexts::LastTraining last;
    last.dataset = run_dataset_name_;
    last.samples = run_samples_->size();
    last.loss = run_->last_loss();
    last.file = run_file_name_;
    run_.reset();
    texts_.set_training(false);

    // The training set is scored once, now; a model loaded later is scored on MNIST test only.
    texts_.set_last_training(last);
    set_loaded_model(std::make_shared<const Mlp>(std::move(trained)), last.file, std::move(run_samples_));

    // The file is written on another thread and the model scored in slices; report once both are done.
    reporting_ = true;
    if (store_.available()) {
        save_ = std::async(std::launch::async,
                [model = model_, path = store_.path_for(last.file)] { return model->save(path); });
    } else {
        last.save_error = tr_("err.no_user_data");
        texts_.set_last_training(last);
    }
    status_.say("status.saving");
}

void Recognizer::set_loaded_model(std::shared_ptr<const Mlp> model, const std::string& name,
        std::shared_ptr<const LabeledImages> train_set) {
    model_ = std::move(model);
    model_name_ = name;
    view_.modelLoaded = true;
    view_.modelMissing = false;
    texts_.set_model({.name = model_name_,
            .architecture = model_->describe(),
            .hidden = model_->hidden(),
            .epochs = model_->epochs_trained()});
    neuron_panel_.set_model(model_);
    run_prediction();
    start_evaluation(std::move(train_set));
}

void Recognizer::start_evaluation(std::shared_ptr<const LabeledImages> train_set) {
    start_dreaming();
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

// One background task per digit, all reading the same model; each picture appears as soon as it is ready.
void Recognizer::start_dreaming() {
    stop_dreaming();
    view_.prototypePaint.reset();
    dream_cancel_ = std::make_shared<std::atomic<bool>>(false);
    for (std::size_t digit = 0; digit < kNumClasses; ++digit) {
        dreams_[digit] = std::async(std::launch::async,
                [model = model_, cancel = dream_cancel_, digit] { return dream_digit(*model, digit, *cancel); });
    }
}

void Recognizer::stop_dreaming() {
    if (dream_cancel_) {
        dream_cancel_->store(true);
    }
    for (auto& dream : dreams_) {
        if (dream.valid()) {
            dream.wait();  // returns quickly: the task checks the flag every step
            dream = {};
        }
    }
}

void Recognizer::collect_dreams() {
    for (std::size_t digit = 0; digit < kNumClasses; ++digit) {
        auto& dream = dreams_[digit];
        if (dream.valid() && dream.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
            if (const auto pixels = dream.get()) {
                view_.prototypePaint.set(digit, *pixels);
            }
        }
    }
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
        rebuild_model_list(last.file);
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
    loading_name_ = model_names_[*selected_];
    model_load_ = std::async(std::launch::async, [path = store_.path_for(loading_name_)] { return Mlp::load(path); });
    status_.say("status.model_loading", {{"name", loading_name_}});
}

void Recognizer::finish_model_load() {
    auto loaded = model_load_.get();
    if (!loaded) {
        status_.say("status.model_load_failed", {{"detail", loaded.error()}});
        return;
    }
    Mlp model = std::move(*loaded);
    if (model.epochs_trained() == 0) {
        model.set_epochs_trained(epochs_in_name(loading_name_));
    }
    set_loaded_model(std::make_shared<const Mlp>(std::move(model)), loading_name_);
    status_.say("status.model_loaded", {{"name", loading_name_}, {"arch", model_->describe()}});
}

void Recognizer::delete_model() {
    const std::string name = model_names_[*selected_];
    if (!store_.remove(name)) {
        status_.say("status.model_delete_failed", {{"name", name}});
        return;
    }
    // The next model in the list takes the selection; the one in memory keeps predicting until another is loaded.
    const std::size_t next = *selected_;
    rebuild_model_list(next + 1 < model_names_.size() ? model_names_[next + 1] : std::string{});
    status_.say("status.model_deleted", {{"name", name}});
}

void Recognizer::refresh_models() {
    const std::string keep = selected_ ? model_names_[*selected_] : std::string{};
    rebuild_model_list(keep);
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

void Recognizer::epochs_up() {
    view_.epochsText = std::to_string(++epochs_);
}

void Recognizer::epochs_down() {
    view_.epochsText = std::to_string(--epochs_);
}

void Recognizer::layers_up() {
    ++layers_;
    show_architecture();
}

void Recognizer::layers_down() {
    --layers_;
    show_architecture();
}

void Recognizer::neurons_up() {
    neurons_ = std::min(neurons_ * 2, kMaxNeurons);
    show_architecture();
}

void Recognizer::neurons_down() {
    neurons_ = std::max(neurons_ / 2, kMinNeurons);
    show_architecture();
}

void Recognizer::show_architecture() {
    view_.layersText = std::to_string(layers_);
    view_.neuronsText = layers_ == 0 ? std::string{"-"} : std::to_string(neurons_);
}

void Recognizer::select_model(std::size_t index) {
    selected_ = index;
    update_model_labels();
}

void Recognizer::run_prediction() {
    const Pixels& pixels = view_.canvasPaint.pixels();
    Mlp::Workspace workspace = model_->make_workspace();
    model_->forward(pixels, workspace);
    Probabilities probs{};
    std::ranges::copy(workspace.activations.back(), probs.begin());
    view_.probabilityPaint.set(probs);
    texts_.set_prediction(static_cast<std::size_t>(std::ranges::max_element(probs) - probs.begin()));

    if (std::ranges::all_of(pixels, [](std::uint8_t value) { return value == 0; })) {
        neuron_panel_.clear_drawing();
    } else {
        neuron_panel_.set_drawing(pixels, workspace);
    }
}

void Recognizer::rebuild_model_list(const std::string& select_name) {
    model_names_ = store_.list();
    selected_.reset();
    if (const auto found = std::ranges::find(model_names_, select_name); found != model_names_.end()) {
        selected_ = static_cast<std::size_t>(found - model_names_.begin());
    } else if (!model_names_.empty()) {
        selected_ = 0;
    }

    model_items_.clear();
    for (std::size_t i = 0; i < model_names_.size(); ++i) {
        auto item = std::make_shared<ListItem>();
        item->select = [this, i] { select_model(i); };
        model_items_.push_back(std::move(item));
    }
    update_model_labels();
    view_.models.set(model_items_);
}

void Recognizer::update_model_labels() {
    for (std::size_t i = 0; i < model_items_.size(); ++i) {
        model_items_[i]->label = (selected_ == i ? "> " : "   ") + model_names_[i];
    }
}

}
