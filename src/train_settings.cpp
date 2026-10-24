#include "train_settings.h"

#include <algorithm>
#include <string>

namespace game {

TrainSettings::TrainSettings(RecognizerViewModel& view)
    : view_(view) {
    view_.epochsUp.bind_to<TrainSettings, &TrainSettings::epochs_up, &TrainSettings::can_epochs_up>(*this);
    view_.epochsDown.bind_to<TrainSettings, &TrainSettings::epochs_down, &TrainSettings::can_epochs_down>(*this);
    view_.layersUp.bind_to<TrainSettings, &TrainSettings::layers_up, &TrainSettings::can_layers_up>(*this);
    view_.layersDown.bind_to<TrainSettings, &TrainSettings::layers_down, &TrainSettings::can_layers_down>(*this);
    view_.neuronsUp.bind_to<TrainSettings, &TrainSettings::neurons_up, &TrainSettings::can_neurons_up>(*this);
    view_.neuronsDown.bind_to<TrainSettings, &TrainSettings::neurons_down, &TrainSettings::can_neurons_down>(*this);
    show_epochs();
    show_architecture();
}

std::vector<std::size_t> TrainSettings::hidden_layers() const {
    return std::vector<std::size_t>(layers_, neurons_);
}

Augmentation TrainSettings::augmentation() const {
    return Augmentation{.shift = view_.shiftAugment.get(), .scale = view_.scaleAugment.get()};
}

void TrainSettings::epochs_up() {
    ++epochs_;
    show_epochs();
}

void TrainSettings::epochs_down() {
    --epochs_;
    show_epochs();
}

void TrainSettings::layers_up() {
    ++layers_;
    show_architecture();
}

void TrainSettings::layers_down() {
    --layers_;
    show_architecture();
}

void TrainSettings::neurons_up() {
    neurons_ = std::min(neurons_ * 2, kMaxNeurons);
    show_architecture();
}

void TrainSettings::neurons_down() {
    neurons_ = std::max(neurons_ / 2, kMinNeurons);
    show_architecture();
}

void TrainSettings::show_epochs() {
    view_.epochsText = std::to_string(epochs_);
}

void TrainSettings::show_architecture() {
    view_.layersText = std::to_string(layers_);
    view_.neuronsText = layers_ == 0 ? std::string{"-"} : std::to_string(neurons_);
}

}
