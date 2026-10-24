#pragma once

#include "model/augment.h"
#include "model/mlp.h"
#include "model/labeled_images.h"
#include "model/parallel.h"

#include <chrono>
#include <memory>
#include <random>
#include <vector>

namespace game {

// Mini-batch SGD over the training set, advanced in slices so the caller can keep its frame rate: each `step` works
// for a time budget and returns. Every batch is split across `parallel`. With `augmentation`, each sample is shown
// scaled and/or moved by random amounts that keep the whole digit inside the frame, new ones every epoch.
class TrainingRun {
public:
    static constexpr std::size_t kBatchSize = 32;
    static constexpr float kLearningRate = 0.1f;

    TrainingRun(Mlp model, std::shared_ptr<const LabeledImages> samples, unsigned epochs, Augmentation augmentation,
            Parallel parallel);

    // Trains for about `budget` (at least one batch). Returns true once every epoch is done.
    bool step(std::chrono::steady_clock::duration budget);

    [[nodiscard]] unsigned epochs() const { return epochs_; }
    [[nodiscard]] unsigned completed_epochs() const { return completed_epochs_; }
    // Mean loss over the samples of the last finished epoch, measured while training on them; 0 before the first.
    [[nodiscard]] float last_loss() const { return last_loss_; }
    // Share of the current epoch's batches done, 0..1.
    [[nodiscard]] float epoch_progress() const;

    [[nodiscard]] Mlp take_model() { return std::move(model_); }

private:
    void run_batch();
    void finish_epoch();

    Mlp model_;
    std::shared_ptr<const LabeledImages> samples_;
    unsigned epochs_;
    Augmentation augmentation_;
    Parallel parallel_;

    std::vector<std::size_t> order_;
    std::size_t cursor_ = 0;
    unsigned completed_epochs_ = 0;
    float last_loss_ = 0.0f;
    std::vector<std::vector<float>> gradients_;  // one buffer per range of the batch
    std::vector<double> range_losses_;
    double epoch_loss_sum_ = 0.0;
    std::uint64_t shift_seed_ = 0;  // new every epoch
    std::mt19937 rng_{std::random_device{}()};
};

}
