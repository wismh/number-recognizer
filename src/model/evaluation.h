#pragma once

#include "model/labeled_images.h"
#include "model/mlp.h"
#include "model/parallel.h"

#include <array>
#include <chrono>
#include <memory>
#include <vector>

namespace game {

// Accuracy of a model on one or more sets of samples, one set after the other, advanced in slices like TrainingRun so a
// frame never waits for it. Every slice is split across `parallel`. Each set is scored twice: as it is, and "mutated",
// with every digit scaled and moved by random amounts (the same ones on every run), which says how well the model
// copes with digits that are not centred or not the usual size.
class Evaluation {
public:
    static constexpr std::size_t kSliceSamples = 2048;

    Evaluation(std::shared_ptr<const Mlp> model, std::vector<std::shared_ptr<const LabeledImages>> sets,
            Parallel parallel);

    // Works for about `budget` (at least one slice). Returns true once every set is scored.
    bool step(std::chrono::steady_clock::duration budget);

    [[nodiscard]] float progress() const;
    // Fractions in [0, 1] for the set at `set` (the order they were passed in), valid once `step` returned true.
    [[nodiscard]] float accuracy(std::size_t set) const;
    [[nodiscard]] float mutated_accuracy(std::size_t set) const;
    // How many samples of `digit` the set at `set` has, and how many of them the model got right (as they are).
    [[nodiscard]] std::size_t digit_total(std::size_t set, std::size_t digit) const;
    [[nodiscard]] std::size_t digit_correct(std::size_t set, std::size_t digit) const;

private:
    struct Stage {
        std::shared_ptr<const LabeledImages> samples;
        std::size_t plain = 0;
        std::size_t mutated = 0;
        std::array<std::size_t, kNumClasses> total_by_digit{};
        std::array<std::size_t, kNumClasses> correct_by_digit{};
    };

    [[nodiscard]] bool done() const { return stage_ >= stages_.size(); }
    void run_slice();

    std::shared_ptr<const Mlp> model_;
    Parallel parallel_;
    std::vector<Stage> stages_;
    std::size_t total_samples_ = 0;
    std::size_t finished_samples_ = 0;
    std::size_t stage_ = 0;
    std::size_t cursor_ = 0;
};

}
