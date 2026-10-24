#pragma once

#include "dataset_manager.h"
#include "loc/tr.h"
#include "model/dream.h"
#include "model/evaluation.h"
#include "model/mlp.h"
#include "model/mnist_data.h"
#include "model/model_store.h"
#include "model/parallel.h"
#include "model/training_run.h"
#include "neuron_panel.h"
#include "recognizer_texts.h"
#include "status_line.h"
#include "view/recognizer_view_model.h"

#include <engine/core/job_system.h>

#include <array>
#include <atomic>
#include <expected>
#include <filesystem>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace game {

// Behavior behind the screen: loads MNIST in the background, trains and loads models, keeps the prediction in sync
// with the canvas. The view model holds the state the XML reads; every button is a method here or on DatasetManager.
class Recognizer {
public:
    static constexpr unsigned kMinEpochs = 1;
    static constexpr unsigned kMaxEpochs = 50;
    static constexpr unsigned kMaxLayers = 4;
    static constexpr unsigned kMinNeurons = 4;
    static constexpr unsigned kMaxNeurons = 256;

    Recognizer(RecognizerViewModel& view, const Tr& tr, ModelStore models, DatasetStore datasets,
            std::filesystem::path data_directory, engine::JobSystem& jobs);

    ~Recognizer();

    Recognizer(const Recognizer&) = delete;
    Recognizer& operator=(const Recognizer&) = delete;

    // Once per frame on the main thread: collects the dataset when it is read and advances training by a slice.
    void update();

    void draw_at(float nx, float ny);
    void end_stroke();

    // Writes every text that depends on the language again.
    void apply_locale();

private:
    [[nodiscard]] bool training() const { return run_.has_value(); }
    // Training, reading or writing a model file, or scoring a model: the buttons that start something wait.
    [[nodiscard]] bool busy() const;
    [[nodiscard]] bool has_training_samples() const;
    [[nodiscard]] bool can_train() const;
    [[nodiscard]] bool can_continue() const;
    [[nodiscard]] bool can_load() const;
    [[nodiscard]] bool can_refresh() const { return !busy(); }
    [[nodiscard]] bool can_clear() const { return !training(); }
    [[nodiscard]] bool can_add() const { return datasets_.can_add_canvas_digit(); }
    [[nodiscard]] bool can_epochs_up() const { return !training() && epochs_ < kMaxEpochs; }
    [[nodiscard]] bool can_epochs_down() const { return !training() && epochs_ > kMinEpochs; }
    [[nodiscard]] bool can_layers_up() const { return !training() && layers_ < kMaxLayers; }
    [[nodiscard]] bool can_layers_down() const { return !training() && layers_ > 0; }
    [[nodiscard]] bool can_neurons_up() const { return !training() && layers_ > 0 && neurons_ < kMaxNeurons; }
    [[nodiscard]] bool can_neurons_down() const { return !training() && layers_ > 0 && neurons_ > kMinNeurons; }

    void train();
    void continue_training();
    void start_training(Mlp model, const std::string& label);
    void load_model();
    void delete_model();
    void refresh_models();
    void clear_canvas();
    void add_sample();
    void epochs_up();
    void epochs_down();
    void layers_up();
    void layers_down();
    void neurons_up();
    void neurons_down();
    void select_model(std::size_t index);

    void finish_dataset_load();
    void finish_model_load();
    void finish_training();
    // Makes `model` the one that predicts, shows its details and scores it on MNIST test, and on `train_set` too when
    // that is not null.
    void set_loaded_model(std::shared_ptr<const Mlp> model, const std::string& name,
            std::shared_ptr<const LabeledImages> train_set = nullptr);
    void start_evaluation(std::shared_ptr<const LabeledImages> train_set);
    void finish_evaluation();
    void start_dreaming();
    void stop_dreaming();
    void collect_dreams();
    void finish_save();
    void report_training();
    void show_training_progress();
    void show_architecture();
    void show_digit_errors();
    void run_prediction();
    void rebuild_model_list(const std::string& select_name);
    void update_model_labels();

    [[nodiscard]] std::vector<std::size_t> hidden_layers() const;
    [[nodiscard]] std::shared_ptr<const LabeledImages> mnist_test() const;

    RecognizerViewModel& view_;
    const Tr& tr_;
    ModelStore store_;
    std::filesystem::path data_directory_;
    Parallel parallel_;
    StatusLine status_;
    RecognizerTexts texts_;
    DatasetManager datasets_;
    NeuronPanel neuron_panel_;

    // Shared with the threads that read or write the file; never changed after it is set.
    std::shared_ptr<const Mlp> model_;
    std::string model_name_;
    std::shared_ptr<const MnistData> mnist_;
    unsigned epochs_ = 3;
    unsigned layers_ = 1;
    unsigned neurons_ = 64;
    std::vector<std::string> model_names_;
    std::vector<std::shared_ptr<ListItem>> model_items_;
    std::optional<std::size_t> selected_;

    std::future<std::expected<MnistData, std::string>> dataset_load_;
    std::future<std::expected<Mlp, std::string>> model_load_;
    std::string loading_name_;
    std::optional<TrainingRun> run_;
    std::string run_file_name_;
    std::string run_dataset_name_;
    std::shared_ptr<const LabeledImages> run_samples_;
    bool evaluating_train_set_ = false;
    std::optional<Evaluation> evaluation_;
    // The ideal-digit pictures being worked out for model_, one task per digit.
    std::shared_ptr<std::atomic<bool>> dream_cancel_;
    std::array<std::future<std::optional<Pixels>>, kNumClasses> dreams_;

    // What a finished training waits for before it reports: its file written and its model scored.
    std::future<std::expected<void, std::string>> save_;
    bool reporting_ = false;
    float test_accuracy_ = 0.0f;
};

}
