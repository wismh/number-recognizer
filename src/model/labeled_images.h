#pragma once

#include "model/digits.h"

#include <cstdint>
#include <span>
#include <vector>

namespace game {

// 28x28 images with a digit label each: `images` holds size() * kImageSize bytes, row-major.
struct LabeledImages {
    std::vector<std::uint8_t> images;
    std::vector<std::uint8_t> labels;

    [[nodiscard]] std::size_t size() const { return labels.size(); }
    [[nodiscard]] std::span<const std::uint8_t> image(std::size_t index) const {
        return std::span<const std::uint8_t>(images).subspan(index * kImageSize, kImageSize);
    }

    void add(const Pixels& pixels, std::uint8_t label) {
        images.insert(images.end(), pixels.begin(), pixels.end());
        labels.push_back(label);
    }

    void remove(std::size_t index) {
        const auto first = images.begin() + static_cast<std::ptrdiff_t>(index * kImageSize);
        images.erase(first, first + static_cast<std::ptrdiff_t>(kImageSize));
        labels.erase(labels.begin() + static_cast<std::ptrdiff_t>(index));
    }
};

}
