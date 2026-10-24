#include "training_session.h"

#include "model/model_store.h"

#include <algorithm>
#include <chrono>
#include <format>

namespace game {

namespace {

// How long one frame may spend training before it hands control back to the UI.
constexpr std::chrono::milliseconds kSlice{8};

}

void TrainingSession::start(Mlp model, const std::string& label, std::string dataset,
        std::shared_ptr<const LabeledImages> samples, unsigned epochs, Augmentation augmentation) {
    dataset_ = std::move(dataset);
    samples_ = std::move(samples);
    file_ = ModelStore::new_file_name(label, model.epochs_trained() + epochs);
    run_.emplace(std::move(model), samples_, epochs, augmentation, parallel_);
    show_progress();
}

std::optional<TrainingSession::Result> TrainingSession::update() {
    if (!run_) {
        return std::nullopt;
    }
    if (!run_->step(kSlice)) {
        show_progress();
        return std::nullopt;
    }
    Mlp trained = run_->take_model();
    trained.set_epochs_trained(trained.epochs_trained() + run_->epochs());
    Result result{.model = std::move(trained),
            .dataset = std::move(dataset_),
            .samples = std::move(samples_),
            .loss = run_->last_loss(),
            .file = std::move(file_)};
    run_.reset();
    return result;
}

void TrainingSession::show_progress() {
    const unsigned done = run_->completed_epochs();
    Tr::Args args{{"dataset", dataset_},
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

}
