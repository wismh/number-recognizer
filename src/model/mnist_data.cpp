#include "model/mnist_data.h"

#include "model/digits.h"

#include <format>
#include <fstream>

namespace game {

namespace {

constexpr std::uint32_t kImagesMagic = 2051;
constexpr std::uint32_t kLabelsMagic = 2049;

bool read_be32(std::istream& in, std::uint32_t& value) {
    unsigned char bytes[4];
    if (!in.read(reinterpret_cast<char*>(bytes), 4)) {
        return false;
    }
    value = (std::uint32_t{bytes[0]} << 24) | (std::uint32_t{bytes[1]} << 16) | (std::uint32_t{bytes[2]} << 8)
            | std::uint32_t{bytes[3]};
    return true;
}

std::expected<std::vector<std::uint8_t>, std::string> read_idx(
        const std::filesystem::path& path, std::uint32_t expected_magic, std::size_t item_size) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return std::unexpected(std::format("cannot open {}", path.string()));
    }

    std::uint32_t magic = 0;
    std::uint32_t count = 0;
    if (!read_be32(in, magic) || magic != expected_magic || !read_be32(in, count)) {
        return std::unexpected(std::format("bad IDX header: {}", path.string()));
    }
    if (expected_magic == kImagesMagic) {
        std::uint32_t rows = 0;
        std::uint32_t cols = 0;
        if (!read_be32(in, rows) || !read_be32(in, cols) || rows * cols != item_size) {
            return std::unexpected(std::format("unexpected image size: {}", path.string()));
        }
    }

    std::vector<std::uint8_t> data(static_cast<std::size_t>(count) * item_size);
    if (!in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()))) {
        return std::unexpected(std::format("truncated IDX file: {}", path.string()));
    }
    return data;
}

}

std::expected<MnistData, std::string> MnistData::load(const std::filesystem::path& directory) {
    MnistData data;
    auto train_images = read_idx(directory / "train-images-idx3-ubyte", kImagesMagic, kImageSize);
    auto train_labels = read_idx(directory / "train-labels-idx1-ubyte", kLabelsMagic, 1);
    auto test_images = read_idx(directory / "t10k-images-idx3-ubyte", kImagesMagic, kImageSize);
    auto test_labels = read_idx(directory / "t10k-labels-idx1-ubyte", kLabelsMagic, 1);
    for (const auto* part : {&train_images, &train_labels, &test_images, &test_labels}) {
        if (!*part) {
            return std::unexpected(part->error());
        }
    }
    data.train.images = std::move(*train_images);
    data.train.labels = std::move(*train_labels);
    data.test.images = std::move(*test_images);
    data.test.labels = std::move(*test_labels);
    return data;
}

}
