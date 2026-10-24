#pragma once

#include "model/augment.h"
#include "view/recognizer_view_model.h"

#include <vector>

namespace game {

// The steppers and checkboxes of the Train tab: how many epochs, how many hidden layers of how many neurons, and which
// changes the samples go through. Locked while a training runs.
class TrainSettings {
public:
    static constexpr unsigned kMinEpochs = 1;
    static constexpr unsigned kMaxEpochs = 50;
    static constexpr unsigned kMaxLayers = 4;
    static constexpr unsigned kMinNeurons = 4;
    static constexpr unsigned kMaxNeurons = 256;

    explicit TrainSettings(RecognizerViewModel& view);

    TrainSettings(const TrainSettings&) = delete;
    TrainSettings& operator=(const TrainSettings&) = delete;

    void set_locked(bool locked) { locked_ = locked; }

    [[nodiscard]] unsigned epochs() const { return epochs_; }
    // The sizes of the hidden layers of a new model.
    [[nodiscard]] std::vector<std::size_t> hidden_layers() const;
    [[nodiscard]] Augmentation augmentation() const;

private:
    [[nodiscard]] bool can_epochs_up() const { return !locked_ && epochs_ < kMaxEpochs; }
    [[nodiscard]] bool can_epochs_down() const { return !locked_ && epochs_ > kMinEpochs; }
    [[nodiscard]] bool can_layers_up() const { return !locked_ && layers_ < kMaxLayers; }
    [[nodiscard]] bool can_layers_down() const { return !locked_ && layers_ > 0; }
    [[nodiscard]] bool can_neurons_up() const { return !locked_ && layers_ > 0 && neurons_ < kMaxNeurons; }
    [[nodiscard]] bool can_neurons_down() const { return !locked_ && layers_ > 0 && neurons_ > kMinNeurons; }

    void epochs_up();
    void epochs_down();
    void layers_up();
    void layers_down();
    void neurons_up();
    void neurons_down();
    void show_epochs();
    void show_architecture();

    RecognizerViewModel& view_;
    bool locked_ = false;
    unsigned epochs_ = 3;
    unsigned layers_ = 1;
    unsigned neurons_ = 64;
};

}
