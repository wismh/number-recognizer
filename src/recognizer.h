#pragma once

#include "dataset_manager.h"
#include "digit_dreamer.h"
#include "loc/tr.h"
#include "model/evaluation.h"
#include "model/mlp.h"
#include "model/mnist_data.h"
#include "model/model_store.h"
#include "model/parallel.h"
#include "model_library.h"
#include "neuron_panel.h"
#include "recognizer_texts.h"
#include "status_line.h"
#include "train_settings.h"
#include "training_session.h"
#include "view/recognizer_view_model.h"

#include <expected>
#include <filesystem>
#include <future>
#include <memory>
#include <optional>
#include <string>

namespace game {

// Behavior behind the screen: loads MNIST in the background, trains and loads models, keeps the prediction in sync
// with the canvas. The view model holds the state the XML reads; every button is a method here or on one of the types
// it is made of (TrainSettings, ModelLibrary, DatasetManager, NeuronPanel).
class Recognizer {
public:
    Recognizer(RecognizerViewModel& view, const Tr& tr, ModelStore models, DatasetStore datasets,
            std::filesystem::path data_directory, Parallel parallel);

    Recognizer(const Recognizer&) = delete;
    Recognizer& operator=(const Recognizer&) = delete;

    // Once per frame on the main thread: collects what the background tasks finished and advances training and scoring
    // by a slice.
    void update();

    // Puts a brush stroke on the canvas at a position in 0..1 of it. Call `canvas_changed` when a batch of them is done.
    void draw_at(float nx, float ny);
    // Predicts again from the canvas.
    void canvas_changed();

    // Writes every text that depends on the language again.
    void apply_locale();

private:
    // Training, reading or writing a model file, or scoring a model: the buttons that start something wait.
    [[nodiscard]] bool busy() const;
    [[nodiscard]] bool has_training_samples() const;
    [[nodiscard]] bool can_train() const;
    [[nodiscard]] bool can_continue() const;
    [[nodiscard]] bool can_load() const;
    [[nodiscard]] bool can_refresh() const { return !busy(); }
    [[nodiscard]] bool can_clear() const { return !session_.active(); }
    [[nodiscard]] bool can_add() const { return datasets_.can_add_canvas_digit(); }

    void train();
    void continue_training();
    void start_training(Mlp model, const std::string& label);
    void load_model();
    void delete_model();
    void refresh_models();
    void clear_canvas();
    void add_sample();

    void finish_dataset_load();
    void finish_model_load(ModelLibrary::Loaded loaded);
    void finish_training(TrainingSession::Result result);
    void finish_save();
    // Makes `model` the one that predicts, shows its details and scores it on MNIST test, and on `train_set` too when
    // that is not null.
    void set_loaded_model(std::shared_ptr<const Mlp> model, const std::string& name,
            std::shared_ptr<const LabeledImages> train_set = nullptr);
    void start_evaluation(std::shared_ptr<const LabeledImages> train_set);
    void finish_evaluation();
    void report_training();
    void run_prediction();

    [[nodiscard]] std::shared_ptr<const LabeledImages> mnist_test() const;

    RecognizerViewModel& view_;
    const Tr& tr_;
    std::filesystem::path data_directory_;
    Parallel parallel_;
    StatusLine status_;
    RecognizerTexts texts_;
    DatasetManager datasets_;
    NeuronPanel neuron_panel_;
    TrainSettings settings_;
    ModelLibrary library_;
    DigitDreamer dreamer_;
    TrainingSession session_;

    // Shared with the threads that read or write the file; never changed after it is set.
    std::shared_ptr<const Mlp> model_;
    std::shared_ptr<const MnistData> mnist_;

    std::future<std::expected<MnistData, std::string>> dataset_load_;
    bool evaluating_train_set_ = false;
    std::optional<Evaluation> evaluation_;

    // What a finished training waits for before it reports: its file written and its model scored.
    std::future<std::expected<void, std::string>> save_;
    bool reporting_ = false;
    float test_accuracy_ = 0.0f;
};

}
