#include "model/training_run.h"

#include "model/augment.h"

#include <algorithm>
#include <numeric>

namespace game {

TrainingRun::TrainingRun(Mlp model, std::shared_ptr<const LabeledImages> samples, unsigned epochs,
        Augmentation augmentation, Parallel parallel)
    : model_(std::move(model))
    , samples_(std::move(samples))
    , epochs_(epochs)
    , augmentation_(augmentation)
    , parallel_(std::move(parallel))
    , order_(samples_->size()) {
    std::iota(order_.begin(), order_.end(), std::size_t{0});
    std::ranges::shuffle(order_, rng_);
    shift_seed_ = rng_();
}

bool TrainingRun::step(std::chrono::steady_clock::duration budget) {
    const auto deadline = std::chrono::steady_clock::now() + budget;
    do {
        if (completed_epochs_ >= epochs_) {
            return true;
        }
        run_batch();
        if (cursor_ >= order_.size()) {
            finish_epoch();
        }
    } while (completed_epochs_ < epochs_ && std::chrono::steady_clock::now() < deadline);
    return completed_epochs_ >= epochs_;
}

float TrainingRun::epoch_progress() const {
    return static_cast<float>(cursor_) / static_cast<float>(order_.size());
}

void TrainingRun::run_batch() {
    const std::size_t count = std::min(kBatchSize, order_.size() - cursor_);
    const std::size_t grain = parallel_.grain_for(count);
    const std::size_t ranges = range_count(count, grain);

    gradients_.resize(ranges);
    for (std::vector<float>& gradient : gradients_) {
        gradient.assign(model_.parameter_count(), 0.0f);
    }

    range_losses_.assign(ranges, 0.0);

    const std::size_t first = cursor_;
    parallel_.for_each_range(count, grain, [&](std::size_t begin, std::size_t end) {
        Mlp::Workspace workspace = model_.make_workspace();
        std::vector<float>& gradient = gradients_[begin / grain];
        Pixels scratch_a;
        Pixels scratch_b;
        double loss = 0.0;
        for (std::size_t i = begin; i < end; ++i) {
            const std::size_t sample = order_[first + i];
            const std::span<const std::uint8_t> input = augmented(samples_->image(sample),
                    augmentation_, shift_seed_ + sample, scratch_a, scratch_b);
            loss += model_.accumulate_gradient(input, samples_->labels[sample], gradient, workspace);
        }
        range_losses_[begin / grain] = loss;
    });
    for (const double loss : range_losses_) {
        epoch_loss_sum_ += loss;
    }

    for (std::size_t r = 1; r < ranges; ++r) {
        for (std::size_t i = 0; i < gradients_[0].size(); ++i) {
            gradients_[0][i] += gradients_[r][i];
        }
    }
    model_.apply_gradient(gradients_[0], kLearningRate / static_cast<float>(count));
    cursor_ += count;
}

void TrainingRun::finish_epoch() {
    ++completed_epochs_;
    cursor_ = 0;
    last_loss_ = static_cast<float>(epoch_loss_sum_ / static_cast<double>(order_.size()));
    epoch_loss_sum_ = 0.0;
    std::ranges::shuffle(order_, rng_);
    shift_seed_ = rng_();
}

}
