#include "model/evaluation.h"

#include "model/training_run.h"
#include "test_support.h"

#include <gtest/gtest.h>

#include <chrono>
#include <memory>

namespace game {

namespace {

using namespace std::chrono_literals;

std::shared_ptr<const Mlp> trained_model() {
    auto train = std::make_shared<const LabeledImages>(test::synthetic_set(40, 1));
    TrainingRun run{Mlp{{16}, 1}, train, 4, Augmentation{}, Parallel{}, 1};
    while (!run.step(50ms)) {
    }
    return std::make_shared<const Mlp>(run.take_model());
}

void run_to_end(Evaluation& evaluation) {
    while (!evaluation.step(50ms)) {
    }
}

}

TEST(Evaluation, ScoresATrainedModelAndCountsEachDigit) {
    auto set = std::make_shared<const LabeledImages>(test::synthetic_set(30, 2));
    Evaluation evaluation{trained_model(), {set}, Parallel{}};
    run_to_end(evaluation);

    EXPECT_GT(evaluation.accuracy(0), 0.95f);
    for (std::size_t digit = 0; digit < kNumClasses; ++digit) {
        EXPECT_EQ(evaluation.digit_total(0, digit), 30u);
        EXPECT_LE(evaluation.digit_correct(0, digit), 30u);
    }
    EXPECT_FLOAT_EQ(evaluation.progress(), 1.0f);
}

TEST(Evaluation, SetsAreScoredInTheOrderTheyWerePassed) {
    auto model = trained_model();
    auto good = std::make_shared<const LabeledImages>(test::synthetic_set(10, 2));
    auto shuffled = std::make_shared<LabeledImages>(*good);
    for (auto& label : shuffled->labels) {
        label = static_cast<std::uint8_t>((label + 1) % kNumClasses);
    }
    Evaluation evaluation{model, {good, shuffled}, Parallel{}};
    run_to_end(evaluation);
    EXPECT_GT(evaluation.accuracy(0), 0.9f);
    EXPECT_LT(evaluation.accuracy(1), 0.1f);
}

TEST(Evaluation, MutatedScoreIsTheSameOnEveryRunAndSplit) {
    auto model = trained_model();
    auto set = std::make_shared<const LabeledImages>(test::synthetic_set(30, 3));
    Parallel split;
    split.workers = 3;

    Evaluation a{model, {set}, Parallel{}};
    Evaluation b{model, {set}, split};
    run_to_end(a);
    run_to_end(b);
    EXPECT_FLOAT_EQ(a.mutated_accuracy(0), b.mutated_accuracy(0));
    EXPECT_FLOAT_EQ(a.accuracy(0), b.accuracy(0));
}

TEST(Evaluation, EmptySetsAreSkipped) {
    auto empty = std::make_shared<const LabeledImages>();
    auto set = std::make_shared<const LabeledImages>(test::synthetic_set(5, 2));
    Evaluation evaluation{trained_model(), {empty, set, empty}, Parallel{}};
    run_to_end(evaluation);
    EXPECT_FLOAT_EQ(evaluation.accuracy(0), 0.0f);
    EXPECT_GT(evaluation.accuracy(1), 0.5f);
    EXPECT_FLOAT_EQ(evaluation.accuracy(2), 0.0f);
}

TEST(Evaluation, NothingToScoreIsDoneAtOnce) {
    Evaluation evaluation{trained_model(), {}, Parallel{}};
    EXPECT_TRUE(evaluation.step(1ms));
    EXPECT_FLOAT_EQ(evaluation.progress(), 1.0f);
}

TEST(Evaluation, ProgressGrowsSliceBySlice) {
    auto set = std::make_shared<const LabeledImages>(test::synthetic_set(500, 2));  // 5000 samples: three slices
    Evaluation evaluation{trained_model(), {set}, Parallel{}};
    EXPECT_FLOAT_EQ(evaluation.progress(), 0.0f);
    EXPECT_FALSE(evaluation.step(0ns));
    EXPECT_NEAR(evaluation.progress(), static_cast<float>(Evaluation::kSliceSamples) / 5000.0f, 1e-5f);
}

}
