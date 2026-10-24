#include "model/evaluation.h"

#include "model/augment.h"

#include <algorithm>

namespace game {

namespace {

std::size_t top_class(const std::vector<float>& probabilities) {
    return static_cast<std::size_t>(std::ranges::max_element(probabilities) - probabilities.begin());
}

struct Hits {
    std::size_t plain = 0;
    std::size_t mutated = 0;
    std::array<std::size_t, kNumClasses> total_by_digit{};
    std::array<std::size_t, kNumClasses> correct_by_digit{};
};

}

Evaluation::Evaluation(std::shared_ptr<const Mlp> model, std::vector<std::shared_ptr<const LabeledImages>> sets,
        Parallel parallel)
    : model_(std::move(model))
    , parallel_(std::move(parallel)) {
    for (auto& samples : sets) {
        total_samples_ += samples->size();
        stages_.push_back(Stage{.samples = std::move(samples)});
    }
    // Sets with no samples have nothing to score.
    while (!done() && stages_[stage_].samples->size() == 0) {
        ++stage_;
    }
}

bool Evaluation::step(std::chrono::steady_clock::duration budget) {
    const auto deadline = std::chrono::steady_clock::now() + budget;
    do {
        if (done()) {
            return true;
        }
        run_slice();
    } while (!done() && std::chrono::steady_clock::now() < deadline);
    return done();
}

float Evaluation::progress() const {
    return total_samples_ == 0 ? 1.0f
                               : static_cast<float>(finished_samples_) / static_cast<float>(total_samples_);
}

float Evaluation::accuracy(std::size_t set) const {
    const Stage& stage = stages_[set];
    return stage.samples->size() == 0 ? 0.0f
                                      : static_cast<float>(stage.plain) / static_cast<float>(stage.samples->size());
}

float Evaluation::mutated_accuracy(std::size_t set) const {
    const Stage& stage = stages_[set];
    return stage.samples->size() == 0 ? 0.0f
                                      : static_cast<float>(stage.mutated) / static_cast<float>(stage.samples->size());
}

std::size_t Evaluation::digit_total(std::size_t set, std::size_t digit) const {
    return stages_[set].total_by_digit[digit];
}

std::size_t Evaluation::digit_correct(std::size_t set, std::size_t digit) const {
    return stages_[set].correct_by_digit[digit];
}

void Evaluation::run_slice() {
    Stage& stage = stages_[stage_];
    const LabeledImages& samples = *stage.samples;
    const std::size_t count = std::min(kSliceSamples, samples.size() - cursor_);
    const std::size_t grain = parallel_.grain_for(count);
    std::vector<Hits> hits(range_count(count, grain));

    const std::size_t first = cursor_;
    parallel_.for_each_range(count, grain, [&](std::size_t begin, std::size_t end) {
        Mlp::Workspace workspace = model_->make_workspace();
        Pixels scratch_a;
        Pixels scratch_b;
        Hits local;
        for (std::size_t i = first + begin; i < first + end; ++i) {
            const std::span<const std::uint8_t> image = samples.image(i);

            model_->forward(image, workspace);
            const bool right = top_class(workspace.activations.back()) == samples.labels[i];
            local.plain += right ? 1 : 0;
            ++local.total_by_digit[samples.labels[i]];
            local.correct_by_digit[samples.labels[i]] += right ? 1 : 0;

            // The changes depend only on the sample's index, so a model scores the same on every run.
            const std::span<const std::uint8_t> mutated =
                    augmented(image, Augmentation{.shift = true, .scale = true}, mix_bits(i), scratch_a, scratch_b);
            model_->forward(mutated, workspace);
            local.mutated += top_class(workspace.activations.back()) == samples.labels[i] ? 1 : 0;
        }
        hits[begin / grain] = local;
    });

    for (const Hits& range : hits) {
        stage.plain += range.plain;
        stage.mutated += range.mutated;
        for (std::size_t digit = 0; digit < kNumClasses; ++digit) {
            stage.total_by_digit[digit] += range.total_by_digit[digit];
            stage.correct_by_digit[digit] += range.correct_by_digit[digit];
        }
    }
    cursor_ += count;
    finished_samples_ += count;
    if (cursor_ >= samples.size()) {
        ++stage_;
        cursor_ = 0;
        while (!done() && stages_[stage_].samples->size() == 0) {
            ++stage_;
        }
    }
}

}
