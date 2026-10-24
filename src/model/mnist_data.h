#pragma once

#include "model/labeled_images.h"

#include <expected>
#include <filesystem>
#include <string>

namespace game {

// MNIST in the original IDX files: images are 28x28 bytes row-major, labels one byte each.
struct MnistData {
    LabeledImages train;  // 60 000 samples
    LabeledImages test;   // 10 000 samples

    // Reads train-images-idx3-ubyte, train-labels-idx1-ubyte, t10k-images-idx3-ubyte, t10k-labels-idx1-ubyte.
    [[nodiscard]] static std::expected<MnistData, std::string> load(const std::filesystem::path& directory);
};

}
