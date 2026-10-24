#include "model/model_store.h"

#include "test_support.h"

#include <gtest/gtest.h>

namespace game {

TEST(ModelStore, ListsOnlyBinFilesSorted) {
    const test::TempDir dir;
    test::write_bytes(dir / "b.bin", {1});
    test::write_bytes(dir / "a.bin", {1});
    test::write_bytes(dir / "note.txt", {1});
    const ModelStore store{dir.path()};
    EXPECT_EQ(store.list(), (std::vector<std::string>{"a.bin", "b.bin"}));
}

TEST(ModelStore, RemoveDeletesTheFile) {
    const test::TempDir dir;
    test::write_bytes(dir / "a.bin", {1});
    const ModelStore store{dir.path()};
    EXPECT_TRUE(store.remove("a.bin"));
    EXPECT_FALSE(store.remove("a.bin"));
    EXPECT_TRUE(store.list().empty());
}

TEST(ModelStore, WithoutADirectoryThereIsNoStorage) {
    const ModelStore store{std::filesystem::path{}};
    EXPECT_FALSE(store.available());
    EXPECT_TRUE(store.list().empty());
    EXPECT_FALSE(store.remove("a.bin"));
}

TEST(ModelStore, NamesCannotLeaveTheDirectory) {
    const test::TempDir dir;
    test::write_bytes(dir / "outside.bin", {1});
    std::filesystem::create_directory(dir / "models");
    const ModelStore store{dir / "models"};
    EXPECT_FALSE(store.remove("../outside.bin"));
    EXPECT_TRUE(std::filesystem::exists(dir / "outside.bin"));
}

TEST(ModelStore, NewFileNameCarriesArchitectureAndEpochs) {
    const std::string name = ModelStore::new_file_name("2x64", 7);
    EXPECT_TRUE(name.starts_with("model_2x64_7ep_"));
    EXPECT_TRUE(name.ends_with(".bin"));
}

TEST(ModelStore, ReadsEpochsFromOldFileNames) {
    EXPECT_EQ(ModelStore::epochs_in_name("model_2x64_12ep_1700000000.bin"), 12u);
    EXPECT_EQ(ModelStore::epochs_in_name("model_linear_3ep_1.bin"), 3u);
    EXPECT_EQ(ModelStore::epochs_in_name("model_linear.bin"), 0u);
    EXPECT_EQ(ModelStore::epochs_in_name("model_linear_ep_1.bin"), 0u);
}

TEST(ModelStore, AbsurdEpochCountInANameIsNotACrash) {
    EXPECT_EQ(ModelStore::epochs_in_name("model_" + std::string(40, '9') + "ep_1.bin"), 0u);
}

}
