#include "model/augment.h"

#include "test_support.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>

namespace game {

namespace {

struct Box {
    int min_x = 28;
    int max_x = -1;
    int min_y = 28;
    int max_y = -1;
};

Box box_of(const Pixels& pixels) {
    Box box;
    for (int y = 0; y < 28; ++y) {
        for (int x = 0; x < 28; ++x) {
            if (pixels[static_cast<std::size_t>(y * 28 + x)] > 0) {
                box.min_x = std::min(box.min_x, x);
                box.max_x = std::max(box.max_x, x);
                box.min_y = std::min(box.min_y, y);
                box.max_y = std::max(box.max_y, y);
            }
        }
    }
    return box;
}

int total(const Pixels& pixels) {
    return std::accumulate(pixels.begin(), pixels.end(), 0);
}

}

TEST(RandomShift, KeepsEveryLitPixelInsideAndPreservesTheDigit) {
    const Pixels digit = test::square(10, 8, 6);
    Pixels out;
    for (std::uint64_t seed = 0; seed < 200; ++seed) {
        random_shift(digit, mix_bits(seed), out);
        EXPECT_EQ(total(out), total(digit));
        const Box box = box_of(out);
        EXPECT_GE(box.min_x, 0);
        EXPECT_LE(box.max_x, 27);
        EXPECT_GE(box.min_y, 0);
        EXPECT_LE(box.max_y, 27);
        EXPECT_EQ(box.max_x - box.min_x, 5);
    }
}

TEST(RandomShift, ReachesBothEdges) {
    const Pixels digit = test::square(10, 10, 4);
    bool touched_left = false;
    bool touched_right = false;
    Pixels out;
    for (std::uint64_t seed = 0; seed < 2000; ++seed) {
        random_shift(digit, mix_bits(seed), out);
        const Box box = box_of(out);
        touched_left |= box.min_x == 0;
        touched_right |= box.max_x == 27;
    }
    EXPECT_TRUE(touched_left);
    EXPECT_TRUE(touched_right);
}

TEST(RandomShift, BlankImageStaysBlank) {
    const Pixels blank{};
    Pixels out = test::square(0, 0, 3);
    random_shift(blank, 123, out);
    EXPECT_EQ(total(out), 0);
}

TEST(RandomScale, DigitAlwaysFitsTheFrame) {
    const Pixels big = test::square(2, 2, 24);
    Pixels out;
    for (std::uint64_t seed = 0; seed < 200; ++seed) {
        random_scale(big, mix_bits(seed), out);
        const Box box = box_of(out);
        EXPECT_GE(box.min_x, 0);
        EXPECT_LE(box.max_x, 27);
        EXPECT_GT(total(out), 0);
    }
}

TEST(RandomScale, BlankImageIsCopied) {
    const Pixels blank{};
    Pixels out = test::square(0, 0, 3);
    random_scale(blank, 5, out);
    EXPECT_EQ(total(out), 0);
}

TEST(Augmented, ReturnsTheImageItselfWhenNothingIsEnabled) {
    const Pixels digit = test::square(5, 5, 5);
    Pixels a;
    Pixels b;
    const auto result = augmented(digit, Augmentation{}, 7, a, b);
    EXPECT_EQ(result.data(), digit.data());
}

TEST(Augmented, SameRandomGivesSameResult) {
    const Pixels digit = test::square(5, 5, 8);
    Pixels a1;
    Pixels b1;
    Pixels a2;
    Pixels b2;
    const Augmentation both{.shift = true, .scale = true};
    const auto first = augmented(digit, both, 42, a1, b1);
    const auto second = augmented(digit, both, 42, a2, b2);
    EXPECT_TRUE(std::ranges::equal(first, second));
}

TEST(MixBits, SpreadsNeighbouringSeeds) {
    EXPECT_NE(mix_bits(1), mix_bits(2));
    EXPECT_NE(mix_bits(0), 0u);
}

}
