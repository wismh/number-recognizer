#include "model/parallel.h"

#include <gtest/gtest.h>

#include <vector>

namespace game {

TEST(Parallel, GrainSplitsAmongWorkers) {
    Parallel parallel;
    parallel.workers = 4;
    EXPECT_EQ(parallel.grain_for(100), 25u);
    EXPECT_EQ(parallel.grain_for(101), 26u);
    EXPECT_EQ(parallel.grain_for(0), 1u);
    EXPECT_EQ(parallel.grain_for(3), 1u);
}

TEST(Parallel, RangeCountRoundsUp) {
    EXPECT_EQ(range_count(10, 4), 3u);
    EXPECT_EQ(range_count(8, 4), 2u);
    EXPECT_EQ(range_count(0, 4), 0u);
}

TEST(Parallel, WithoutRunnerCoversEveryIndexOnce) {
    const Parallel parallel;
    std::vector<int> seen(10, 0);
    parallel.for_each_range(10, 3, [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            ++seen[i];
        }
    });
    EXPECT_EQ(seen, std::vector<int>(10, 1));
}

TEST(Parallel, EmptyCountCallsNothing) {
    const Parallel parallel;
    bool called = false;
    parallel.for_each_range(0, 1, [&](std::size_t, std::size_t) { called = true; });
    EXPECT_FALSE(called);
}

}
