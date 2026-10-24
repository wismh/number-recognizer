#include "model/training_run.h"

#include "model/evaluation.h"
#include "test_support.h"

#include <gtest/gtest.h>

#include <chrono>
#include <memory>

namespace game {

namespace {

using namespace std::chrono_literals;

void run_to_end(TrainingRun& run) {
    while (!run.step(50ms)) {
    }
}

float accuracy_of(const Mlp& model, const std::shared_ptr<const LabeledImages>& set) {
    Evaluation evaluation{std::make_shared<const Mlp>(model), {set}, Parallel{}};
    while (!evaluation.step(50ms)) {
    }
    return evaluation.accuracy(0);
}

}

TEST(TrainingRun, LearnsAnEasyProblem) {
    auto train = std::make_shared<const LabeledImages>(test::synthetic_set(40, 1));
    auto check = std::make_shared<const LabeledImages>(test::synthetic_set(10, 2));

    TrainingRun run{Mlp{{16}, 1}, train, 4, Augmentation{}, Parallel{}, 1};
    run_to_end(run);
    EXPECT_EQ(run.completed_epochs(), 4u);
    EXPECT_GT(run.last_loss(), 0.0f);
    EXPECT_LT(run.last_loss(), 1.0f);

    const Mlp trained = run.take_model();
    EXPECT_GT(accuracy_of(trained, check), 0.95f);
}

TEST(TrainingRun, LossFallsFromEpochToEpoch) {
    auto train = std::make_shared<const LabeledImages>(test::synthetic_set(40, 1));
    TrainingRun run{Mlp{{}, 1}, train, 1, Augmentation{}, Parallel{}, 1};
    run_to_end(run);
    const float first = run.last_loss();

    TrainingRun longer{Mlp{{}, 1}, train, 5, Augmentation{}, Parallel{}, 1};
    run_to_end(longer);
    EXPECT_LT(longer.last_loss(), first);
}

TEST(TrainingRun, SameSeedGivesTheSameModel) {
    auto train = std::make_shared<const LabeledImages>(test::synthetic_set(10, 1));
    const Pixels image = test::square(3, 3, 5);

    TrainingRun a{Mlp{{8}, 5}, train, 2, Augmentation{.shift = true}, Parallel{}, 9};
    TrainingRun b{Mlp{{8}, 5}, train, 2, Augmentation{.shift = true}, Parallel{}, 9};
    run_to_end(a);
    run_to_end(b);
    EXPECT_EQ(a.take_model().probabilities(image), b.take_model().probabilities(image));
}

TEST(TrainingRun, SplittingABatchAcrossRangesGivesTheSameModel) {
    auto train = std::make_shared<const LabeledImages>(test::synthetic_set(10, 1));
    const Pixels image = test::square(3, 3, 5);

    Parallel split;
    split.workers = 4;
    TrainingRun serial{Mlp{{8}, 5}, train, 1, Augmentation{}, Parallel{}, 9};
    TrainingRun ranged{Mlp{{8}, 5}, train, 1, Augmentation{}, split, 9};
    run_to_end(serial);
    run_to_end(ranged);
    const Probabilities a = serial.take_model().probabilities(image);
    const Probabilities b = ranged.take_model().probabilities(image);
    for (std::size_t i = 0; i < kNumClasses; ++i) {
        EXPECT_NEAR(a[i], b[i], 1e-4f);
    }
}

TEST(TrainingRun, ProgressGoesFromZeroToOneWithinAnEpoch) {
    auto train = std::make_shared<const LabeledImages>(test::synthetic_set(20, 1));
    TrainingRun run{Mlp{{}, 1}, train, 2, Augmentation{}, Parallel{}, 1};
    EXPECT_FLOAT_EQ(run.epoch_progress(), 0.0f);
    EXPECT_FALSE(run.step(0ns));  // at least one batch
    EXPECT_GT(run.epoch_progress(), 0.0f);
    EXPECT_LT(run.epoch_progress(), 1.0f);
}

TEST(TrainingRun, StepDoesNotOvershootTheLastEpoch) {
    auto train = std::make_shared<const LabeledImages>(test::synthetic_set(5, 1));
    TrainingRun run{Mlp{{}, 1}, train, 1, Augmentation{}, Parallel{}, 1};
    run_to_end(run);
    EXPECT_TRUE(run.step(10ms));
    EXPECT_EQ(run.completed_epochs(), 1u);
}

TEST(TrainingRun, EmptyDatasetFinishesWithoutTraining) {
    auto empty = std::make_shared<const LabeledImages>();
    TrainingRun run{Mlp{{4}, 1}, empty, 3, Augmentation{}, Parallel{}, 1};
    run_to_end(run);
    EXPECT_EQ(run.completed_epochs(), 3u);
    EXPECT_FLOAT_EQ(run.epoch_progress(), 0.0f);
}

}
