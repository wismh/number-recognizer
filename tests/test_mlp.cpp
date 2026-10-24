#include "model/mlp.h"

#include "test_support.h"

#include <gtest/gtest.h>

#include <cmath>
#include <numeric>

namespace game {

namespace {

float loss_of(const Mlp& model, const Pixels& image, std::size_t label) {
    return -std::log(model.probabilities(image)[label]);
}

Pixels sample_image() {
    return test::square(6, 6, 10, 200);
}

}

TEST(Mlp, SameSeedGivesSameWeights) {
    const Mlp a{{16}, 7};
    const Mlp b{{16}, 7};
    const Mlp c{{16}, 8};
    const Pixels image = sample_image();
    EXPECT_EQ(a.probabilities(image), b.probabilities(image));
    EXPECT_NE(a.probabilities(image), c.probabilities(image));
}

TEST(Mlp, DescribesItsShapeAndCountsParameters) {
    const Mlp model{{64, 32}, 1};
    EXPECT_EQ(model.describe(), "784-64-32-10");
    EXPECT_EQ(model.parameter_count(), 784u * 64 + 64 + 64u * 32 + 32 + 32u * 10 + 10);
    EXPECT_EQ(model.layer_count(), 3u);
    EXPECT_EQ(Mlp({}, 1).describe(), "784-10");
}

TEST(Mlp, LabelNamesTheArchitectureForFileNames) {
    EXPECT_EQ(Mlp({}, 1).label(), "linear");
    EXPECT_EQ(Mlp({64}, 1).label(), "1x64");
    EXPECT_EQ(Mlp({32, 32, 32}, 1).label(), "3x32");
    EXPECT_EQ(Mlp({64, 32}, 1).label(), "64-32");
}

TEST(Mlp, ProbabilitiesAreADistribution) {
    for (const auto& hidden : {std::vector<std::size_t>{}, std::vector<std::size_t>{8}, std::vector<std::size_t>{8, 8}}) {
        const Mlp model{hidden, 3};
        const Probabilities probs = model.probabilities(sample_image());
        EXPECT_NEAR(std::accumulate(probs.begin(), probs.end(), 0.0f), 1.0f, 1e-5f);
        for (const float p : probs) {
            EXPECT_GE(p, 0.0f);
            EXPECT_LE(p, 1.0f);
        }
    }
}

TEST(Mlp, PredictIsTheMostLikelyDigit) {
    const Mlp model{{8}, 5};
    const Pixels image = sample_image();
    const Probabilities probs = model.probabilities(image);
    EXPECT_EQ(model.predict(image), static_cast<std::size_t>(std::ranges::max_element(probs) - probs.begin()));
}

TEST(Mlp, WeightGradientPointsDownhill) {
    // Stepping along -gradient by a small eps lowers the loss by about eps * |gradient|^2.
    for (const auto& hidden : {std::vector<std::size_t>{}, std::vector<std::size_t>{12, 10}}) {
        Mlp model{hidden, 11};
        const Pixels image = sample_image();
        constexpr std::size_t kLabel = 4;
        std::vector<float> gradient(model.parameter_count(), 0.0f);
        Mlp::Workspace workspace = model.make_workspace();
        const float loss = model.accumulate_gradient(image, kLabel, gradient, workspace);
        EXPECT_NEAR(loss, loss_of(model, image, kLabel), 1e-5f);

        double norm = 0.0;
        for (const float g : gradient) {
            norm += static_cast<double>(g) * g;
        }
        constexpr float kEps = 1e-3f;
        model.apply_gradient(gradient, kEps);
        const double drop = loss - loss_of(model, image, kLabel);
        EXPECT_NEAR(drop, kEps * norm, 0.1 * kEps * norm) << "hidden layers: " << hidden.size();
    }
}

TEST(Mlp, InputGradientMatchesFiniteDifferences) {
    const Mlp model{{16, 16}, 21};
    std::vector<float> input(kImageSize, 0.3f);
    constexpr std::size_t kLabel = 7;
    Mlp::Workspace workspace = model.make_workspace();
    model.forward_input(input, workspace);
    std::vector<float> gradient;
    model.input_gradient(kLabel, workspace, gradient);
    ASSERT_EQ(gradient.size(), kImageSize);

    const auto loss_at = [&](const std::vector<float>& values) {
        Mlp::Workspace w = model.make_workspace();
        model.forward_input(values, w);
        return -std::log(w.activations.back()[kLabel]);
    };
    constexpr float kStep = 1e-2f;
    for (const std::size_t pixel : {std::size_t{100}, std::size_t{300}, std::size_t{500}}) {
        std::vector<float> up = input;
        std::vector<float> down = input;
        up[pixel] += kStep;
        down[pixel] -= kStep;
        const float numeric = (loss_at(up) - loss_at(down)) / (2.0f * kStep);
        EXPECT_NEAR(gradient[pixel], numeric, 1e-3f + 0.05f * std::abs(numeric)) << "pixel " << pixel;
    }
}

TEST(Mlp, NeuronInputGradientMatchesFiniteDifferences) {
    const Mlp model{{16, 16}, 22};
    std::vector<float> input(kImageSize, 0.4f);
    constexpr std::size_t kLayer = 1;
    constexpr std::size_t kNeuron = 3;
    Mlp::Workspace workspace = model.make_workspace();
    model.forward_input_below(input, kLayer, workspace);
    std::vector<float> gradient;
    model.neuron_input_gradient(kLayer, kNeuron, workspace, gradient);

    // The neuron's weighted sum before ReLU, from the activations of the layer below.
    const auto sum_at = [&](const std::vector<float>& values) {
        Mlp::Workspace w = model.make_workspace();
        model.forward_input_below(values, kLayer, w);
        const std::vector<float>& below = w.activations[kLayer];
        float sum = model.bias(kLayer, kNeuron);
        const auto weights = model.incoming_weights(kLayer, kNeuron);
        for (std::size_t i = 0; i < weights.size(); ++i) {
            sum += weights[i] * below[i];
        }
        return sum;
    };
    constexpr float kStep = 1e-2f;
    for (const std::size_t pixel : {std::size_t{50}, std::size_t{400}}) {
        std::vector<float> up = input;
        std::vector<float> down = input;
        up[pixel] += kStep;
        down[pixel] -= kStep;
        const float numeric = (sum_at(up) - sum_at(down)) / (2.0f * kStep);
        EXPECT_NEAR(gradient[pixel], numeric, 1e-3f + 0.05f * std::abs(numeric)) << "pixel " << pixel;
    }
}

TEST(Mlp, LastHiddenNeuronInfluenceIsItsWeightsToTheDigits) {
    const Mlp model{{8}, 31};
    const Probabilities influence = model.neuron_influence(0, 2, nullptr);
    for (std::size_t digit = 0; digit < kNumClasses; ++digit) {
        EXPECT_FLOAT_EQ(influence[digit], model.incoming_weights(1, digit)[2]);
    }
}

TEST(MlpFile, RoundTripKeepsEverything) {
    const test::TempDir dir;
    Mlp model{{16, 8}, 41};
    model.set_epochs_trained(5);
    ASSERT_TRUE(model.save(dir / "m.bin"));

    const auto loaded = Mlp::load(dir / "m.bin");
    ASSERT_TRUE(loaded) << loaded.error();
    EXPECT_EQ(loaded->hidden(), model.hidden());
    EXPECT_EQ(loaded->epochs_trained(), 5u);
    const Pixels image = sample_image();
    EXPECT_EQ(loaded->probabilities(image), model.probabilities(image));
}

TEST(MlpFile, SaveCreatesMissingDirectoriesAndLeavesNoTemporaryFile) {
    const test::TempDir dir;
    const Mlp model{{4}, 1};
    ASSERT_TRUE(model.save(dir / "a" / "b" / "m.bin"));
    EXPECT_TRUE(std::filesystem::exists(dir / "a" / "b" / "m.bin"));
    EXPECT_FALSE(std::filesystem::exists(dir / "a" / "b" / "m.bin.tmp"));
}

TEST(MlpFile, ReadsTheLinearVersionOne) {
    const test::TempDir dir;
    std::vector<std::uint8_t> bytes{'M', 'N', 'W', 'T'};
    test::append(bytes, std::uint32_t{1});
    test::append(bytes, std::uint32_t{784});
    test::append(bytes, std::uint32_t{10});
    for (std::size_t i = 0; i < kImageSize * kNumClasses + kNumClasses; ++i) {
        test::append(bytes, 0.0f);
    }
    test::write_bytes(dir / "v1.bin", bytes);

    const auto loaded = Mlp::load(dir / "v1.bin");
    ASSERT_TRUE(loaded) << loaded.error();
    EXPECT_TRUE(loaded->hidden().empty());
    EXPECT_EQ(loaded->epochs_trained(), 0u);
}

TEST(MlpFile, RejectsDamagedFiles) {
    const test::TempDir dir;
    EXPECT_FALSE(Mlp::load(dir / "missing.bin"));

    test::write_bytes(dir / "magic.bin", {'X', 'X', 'X', 'X', 0, 0, 0, 0});
    EXPECT_FALSE(Mlp::load(dir / "magic.bin"));

    std::vector<std::uint8_t> version{'M', 'N', 'W', 'T'};
    test::append(version, std::uint32_t{99});
    test::append(version, std::uint32_t{784});
    test::append(version, std::uint32_t{10});
    test::write_bytes(dir / "version.bin", version);
    EXPECT_FALSE(Mlp::load(dir / "version.bin"));

    std::vector<std::uint8_t> shape{'M', 'N', 'W', 'T'};
    test::append(shape, std::uint32_t{3});
    test::append(shape, std::uint32_t{100});
    test::append(shape, std::uint32_t{10});
    test::write_bytes(dir / "shape.bin", shape);
    EXPECT_FALSE(Mlp::load(dir / "shape.bin"));
}

TEST(MlpFile, RejectsTruncatedWeights) {
    const test::TempDir dir;
    const Mlp model{{8}, 1};
    ASSERT_TRUE(model.save(dir / "m.bin"));
    std::filesystem::resize_file(dir / "m.bin", std::filesystem::file_size(dir / "m.bin") - 4);
    EXPECT_FALSE(Mlp::load(dir / "m.bin"));
}

TEST(MlpFile, RejectsAbsurdLayerSizes) {
    const test::TempDir dir;
    std::vector<std::uint8_t> bytes{'M', 'N', 'W', 'T'};
    test::append(bytes, std::uint32_t{3});
    test::append(bytes, std::uint32_t{784});
    test::append(bytes, std::uint32_t{10});
    test::append(bytes, std::uint32_t{1});
    test::append(bytes, std::uint32_t{Mlp::kMaxNeurons + 1});
    test::append(bytes, std::uint32_t{0});
    test::write_bytes(dir / "big.bin", bytes);
    EXPECT_FALSE(Mlp::load(dir / "big.bin"));
}

TEST(MlpFile, FailedSaveKeepsTheOldFile) {
    const test::TempDir dir;
    const Mlp first{{4}, 1};
    ASSERT_TRUE(first.save(dir / "m.bin"));
    const auto size = std::filesystem::file_size(dir / "m.bin");

    // A directory in the place of the temporary file makes the write fail.
    std::filesystem::create_directory(dir / "m.bin.tmp");
    const Mlp second{{4, 4}, 2};
    EXPECT_FALSE(second.save(dir / "m.bin"));
    EXPECT_EQ(std::filesystem::file_size(dir / "m.bin"), size);
}

}
