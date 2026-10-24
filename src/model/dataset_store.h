#pragma once

#include "model/labeled_images.h"

#include <expected>
#include <filesystem>
#include <string>
#include <vector>

namespace game {

// The user's own datasets, one `<name>.dset` file each in one directory. A file holds a magic, a version, the number of
// samples, then per sample one label byte and 784 image bytes. An empty directory path means there is no storage.
class DatasetStore {
public:
    static constexpr std::size_t kMaxNameLength = 32;

    explicit DatasetStore(std::filesystem::path directory);

    [[nodiscard]] bool available() const { return !directory_.empty(); }
    // Names without the extension, sorted.
    [[nodiscard]] std::vector<std::string> list() const;
    [[nodiscard]] std::expected<LabeledImages, std::string> load(const std::string& name) const;
    // Writes the whole file: creates the dataset when it does not exist yet.
    [[nodiscard]] std::expected<void, std::string> save(const std::string& name, const LabeledImages& data) const;
    bool remove(const std::string& name) const;

    enum class NameProblem { None, Empty, TooLong, BadCharacters, Reserved };
    // A name is letters, digits, '-' and '_' only, 1..kMaxNameLength long, and not the reserved name "MNIST" in any
    // case.
    [[nodiscard]] static NameProblem check_name(const std::string& name);

private:
    [[nodiscard]] std::filesystem::path path_for(const std::string& name) const;

    std::filesystem::path directory_;
};

}
