#include "model/normalize.h"

#include "test_support.h"

#include <gtest/gtest.h>

#include <algorithm>

namespace game {

namespace {

struct Mass {
    double x = 0.0;
    double y = 0.0;
    double sum = 0.0;
};

Mass mass_of(const Pixels& pixels) {
    Mass mass;
    for (int y = 0; y < 28; ++y) {
        for (int x = 0; x < 28; ++x) {
            const double v = pixels[static_cast<std::size_t>(y * 28 + x)];
            mass.sum += v;
            mass.x += v * (x + 0.5);
            mass.y += v * (y + 0.5);
        }
    }
    mass.x /= mass.sum;
    mass.y /= mass.sum;
    return mass;
}

}

TEST(NormalizeDigit, BlankImageIsReturnedAsIs) {
    const Pixels blank{};
    EXPECT_EQ(normalize_digit(blank), blank);
}

TEST(NormalizeDigit, LongerSideBecomesTwentyPixels) {
    const Pixels result = normalize_digit(test::square(1, 1, 10));
    int min_x = 28;
    int max_x = -1;
    for (int y = 0; y < 28; ++y) {
        for (int x = 0; x < 28; ++x) {
            if (result[static_cast<std::size_t>(y * 28 + x)] > 128) {
                min_x = std::min(min_x, x);
                max_x = std::max(max_x, x);
            }
        }
    }
    EXPECT_NEAR(max_x - min_x + 1, 20, 1);
}

TEST(NormalizeDigit, CentreOfMassMovesToTheMiddle) {
    const Mass mass = mass_of(normalize_digit(test::square(2, 17, 6)));
    EXPECT_NEAR(mass.x, 14.0, 0.6);
    EXPECT_NEAR(mass.y, 14.0, 0.6);
}

}
