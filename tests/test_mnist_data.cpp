#include "model/mnist_data.h"

#include "test_support.h"

#include <gtest/gtest.h>

namespace game {

namespace {

void write_images(const std::filesystem::path& path, std::uint32_t count, std::uint32_t declared = 0) {
    std::vector<std::uint8_t> bytes;
    test::append_be32(bytes, 2051);
    test::append_be32(bytes, declared == 0 ? count : declared);
    test::append_be32(bytes, 28);
    test::append_be32(bytes, 28);
    for (std::uint32_t i = 0; i < count; ++i) {
        for (std::size_t p = 0; p < kImageSize; ++p) {
            bytes.push_back(static_cast<std::uint8_t>(i));
        }
    }
    test::write_bytes(path, bytes);
}

void write_labels(const std::filesystem::path& path, std::uint32_t count) {
    std::vector<std::uint8_t> bytes;
    test::append_be32(bytes, 2049);
    test::append_be32(bytes, count);
    for (std::uint32_t i = 0; i < count; ++i) {
        bytes.push_back(static_cast<std::uint8_t>(i % 10));
    }
    test::write_bytes(path, bytes);
}

void write_all(const std::filesystem::path& dir, std::uint32_t train, std::uint32_t test) {
    write_images(dir / "train-images-idx3-ubyte", train);
    write_labels(dir / "train-labels-idx1-ubyte", train);
    write_images(dir / "t10k-images-idx3-ubyte", test);
    write_labels(dir / "t10k-labels-idx1-ubyte", test);
}

}

TEST(MnistData, ReadsTheFourIdxFiles) {
    const test::TempDir dir;
    write_all(dir.path(), 12, 5);

    const auto data = MnistData::load(dir.path());
    ASSERT_TRUE(data) << data.error();
    EXPECT_EQ(data->train.size(), 12u);
    EXPECT_EQ(data->test.size(), 5u);
    EXPECT_EQ(data->train.labels[3], 3);
    EXPECT_EQ(data->train.image(7)[0], 7);
    EXPECT_EQ(data->test.image(4).size(), kImageSize);
}

TEST(MnistData, MissingFileIsAnError) {
    const test::TempDir dir;
    write_all(dir.path(), 4, 4);
    std::filesystem::remove(dir / "t10k-labels-idx1-ubyte");
    EXPECT_FALSE(MnistData::load(dir.path()));
}

TEST(MnistData, BadMagicIsAnError) {
    const test::TempDir dir;
    write_all(dir.path(), 4, 4);
    write_labels(dir / "train-images-idx3-ubyte", 4);  // a labels file where images belong
    EXPECT_FALSE(MnistData::load(dir.path()));
}

TEST(MnistData, TruncatedFileIsAnError) {
    const test::TempDir dir;
    write_all(dir.path(), 4, 4);
    std::filesystem::resize_file(dir / "train-images-idx3-ubyte", 100);
    EXPECT_FALSE(MnistData::load(dir.path()));
}

TEST(MnistData, HugeCountInTheHeaderIsAnErrorNotACrash) {
    const test::TempDir dir;
    write_all(dir.path(), 4, 4);
    write_images(dir / "train-images-idx3-ubyte", 4, 0xffffffffu);
    EXPECT_FALSE(MnistData::load(dir.path()));
}

TEST(MnistData, ImageAndLabelCountsMustAgree) {
    const test::TempDir dir;
    write_all(dir.path(), 4, 4);
    write_labels(dir / "train-labels-idx1-ubyte", 6);
    EXPECT_FALSE(MnistData::load(dir.path()));
}

}
