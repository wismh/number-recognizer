#include "model/dataset_store.h"

#include "test_support.h"

#include <gtest/gtest.h>

namespace game {

TEST(DatasetStoreName, AcceptsPlainNames) {
    EXPECT_EQ(DatasetStore::check_name("mine"), DatasetStore::NameProblem::None);
    EXPECT_EQ(DatasetStore::check_name("Digits_2-b"), DatasetStore::NameProblem::None);
    EXPECT_EQ(DatasetStore::check_name(std::string(DatasetStore::kMaxNameLength, 'a')), DatasetStore::NameProblem::None);
}

TEST(DatasetStoreName, RejectsBadNames) {
    EXPECT_EQ(DatasetStore::check_name(""), DatasetStore::NameProblem::Empty);
    EXPECT_EQ(DatasetStore::check_name(std::string(DatasetStore::kMaxNameLength + 1, 'a')),
            DatasetStore::NameProblem::TooLong);
    EXPECT_EQ(DatasetStore::check_name("a b"), DatasetStore::NameProblem::BadCharacters);
    EXPECT_EQ(DatasetStore::check_name("../x"), DatasetStore::NameProblem::BadCharacters);
    EXPECT_EQ(DatasetStore::check_name("a.b"), DatasetStore::NameProblem::BadCharacters);
}

TEST(DatasetStoreName, MnistIsReservedInAnyCase) {
    EXPECT_EQ(DatasetStore::check_name("MNIST"), DatasetStore::NameProblem::Reserved);
    EXPECT_EQ(DatasetStore::check_name("mnist"), DatasetStore::NameProblem::Reserved);
    EXPECT_EQ(DatasetStore::check_name("MnIsT"), DatasetStore::NameProblem::Reserved);
}

TEST(DatasetStore, RoundTripKeepsLabelsAndPixels) {
    const test::TempDir dir;
    const DatasetStore store{dir / "sets"};
    const LabeledImages data = test::synthetic_set(3, 1);
    ASSERT_TRUE(store.save("mine", data));

    const auto loaded = store.load("mine");
    ASSERT_TRUE(loaded) << loaded.error();
    EXPECT_EQ(loaded->labels, data.labels);
    EXPECT_EQ(loaded->images, data.images);
    EXPECT_FALSE(std::filesystem::exists(dir / "sets" / "mine.dset.tmp"));
}

TEST(DatasetStore, ListsNamesSortedAndRemoves) {
    const test::TempDir dir;
    const DatasetStore store{dir.path()};
    ASSERT_TRUE(store.save("b", {}));
    ASSERT_TRUE(store.save("a", {}));
    EXPECT_EQ(store.list(), (std::vector<std::string>{"a", "b"}));
    EXPECT_TRUE(store.remove("a"));
    EXPECT_FALSE(store.remove("a"));
    EXPECT_EQ(store.list(), (std::vector<std::string>{"b"}));
}

TEST(DatasetStore, WithoutADirectoryThereIsNoStorage) {
    const DatasetStore store{std::filesystem::path{}};
    EXPECT_FALSE(store.available());
    EXPECT_TRUE(store.list().empty());
    EXPECT_FALSE(store.save("x", {}));
    EXPECT_FALSE(store.remove("x"));
}

TEST(DatasetStore, RejectsWrongAndDamagedFiles) {
    const test::TempDir dir;
    const DatasetStore store{dir.path()};
    EXPECT_FALSE(store.load("missing"));

    test::write_bytes(dir / "junk.dset", {1, 2, 3});
    EXPECT_FALSE(store.load("junk"));

    ASSERT_TRUE(store.save("cut", test::synthetic_set(2, 1)));
    std::filesystem::resize_file(dir / "cut.dset", std::filesystem::file_size(dir / "cut.dset") - 10);
    EXPECT_FALSE(store.load("cut"));
}

TEST(DatasetStore, RejectsLabelsOutOfRange) {
    const test::TempDir dir;
    const DatasetStore store{dir.path()};
    LabeledImages data = test::synthetic_set(1, 1);
    data.labels[0] = 10;
    ASSERT_TRUE(store.save("bad", data));
    EXPECT_FALSE(store.load("bad"));
}

TEST(DatasetStore, HugeCountInTheHeaderIsAnErrorNotACrash) {
    const test::TempDir dir;
    const DatasetStore store{dir.path()};
    std::vector<std::uint8_t> bytes{'N', 'R', 'D', 'S'};
    test::append(bytes, std::uint32_t{1});
    test::append(bytes, std::uint32_t{0xffffffffu});
    test::write_bytes(dir / "huge.dset", bytes);
    EXPECT_FALSE(store.load("huge"));
}

}
