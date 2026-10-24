#include "model/dataset_store.h"

#include "model/binary_io.h"

#include <algorithm>
#include <cctype>
#include <format>
#include <fstream>

namespace game {

namespace {

constexpr char kMagic[4] = {'N', 'R', 'D', 'S'};
constexpr std::uint32_t kVersion = 1;
constexpr const char* kExtension = ".dset";

}

DatasetStore::DatasetStore(std::filesystem::path directory)
    : directory_(std::move(directory)) {}

std::filesystem::path DatasetStore::path_for(const std::string& name) const {
    return directory_ / (name + kExtension);
}

std::vector<std::string> DatasetStore::list() const {
    std::vector<std::string> names;
    if (!available()) {
        return names;
    }
    std::error_code error;
    for (const auto& entry : std::filesystem::directory_iterator(directory_, error)) {
        if (entry.is_regular_file(error) && entry.path().extension() == kExtension) {
            names.push_back(entry.path().stem().string());
        }
    }
    std::ranges::sort(names);
    return names;
}

std::expected<LabeledImages, std::string> DatasetStore::load(const std::string& name) const {
    std::ifstream in(path_for(name), std::ios::binary);
    if (!in) {
        return std::unexpected(std::format("cannot open dataset {}", name));
    }

    char magic[4]{};
    std::uint32_t version = 0;
    std::uint32_t count = 0;
    in.read(magic, sizeof(magic));
    if (!in || !read_value(in, version) || !read_value(in, count)
            || !std::equal(std::begin(magic), std::end(magic), std::begin(kMagic)) || version != kVersion) {
        return std::unexpected(std::format("{} is not a dataset file", name));
    }

    // The header must agree with the file, or a damaged count would ask for gigabytes.
    std::error_code size_error;
    const std::uintmax_t file_size = std::filesystem::file_size(path_for(name), size_error);
    constexpr std::uintmax_t kHeader = sizeof(kMagic) + 2 * sizeof(std::uint32_t);
    if (size_error || file_size != kHeader + std::uintmax_t{count} * (1 + kImageSize)) {
        return std::unexpected(std::format("dataset {} is damaged", name));
    }

    LabeledImages data;
    data.labels.resize(count);
    data.images.resize(static_cast<std::size_t>(count) * kImageSize);
    for (std::uint32_t i = 0; i < count; ++i) {
        in.read(reinterpret_cast<char*>(&data.labels[i]), 1);
        in.read(reinterpret_cast<char*>(data.images.data() + static_cast<std::size_t>(i) * kImageSize),
                static_cast<std::streamsize>(kImageSize));
    }
    if (!in || std::ranges::any_of(data.labels, [](std::uint8_t label) { return label >= kNumClasses; })) {
        return std::unexpected(std::format("dataset {} is damaged", name));
    }
    return data;
}

std::expected<void, std::string> DatasetStore::save(const std::string& name, const LabeledImages& data) const {
    if (!available()) {
        return std::unexpected("no user data directory");
    }
    std::error_code error;
    std::filesystem::create_directories(directory_, error);

    // Written beside the final name and renamed, so a failed write never leaves half a dataset.
    const std::filesystem::path target = path_for(name);
    const std::filesystem::path temporary = target.string() + ".tmp";
    {
        std::ofstream out(temporary, std::ios::binary);
        if (!out) {
            return std::unexpected(std::format("cannot write dataset {}", name));
        }
        const auto count = static_cast<std::uint32_t>(data.size());
        out.write(kMagic, sizeof(kMagic));
        write_value(out, kVersion);
        write_value(out, count);
        for (std::size_t i = 0; i < data.size(); ++i) {
            out.write(reinterpret_cast<const char*>(&data.labels[i]), 1);
            out.write(reinterpret_cast<const char*>(data.images.data() + i * kImageSize),
                    static_cast<std::streamsize>(kImageSize));
        }
        if (!out) {
            return std::unexpected(std::format("write failed for dataset {}", name));
        }
    }
    std::filesystem::rename(temporary, target, error);
    if (error) {
        return std::unexpected(std::format("cannot replace dataset {}: {}", name, error.message()));
    }
    return {};
}

bool DatasetStore::remove(const std::string& name) const {
    std::error_code error;
    return available() && std::filesystem::remove(path_for(name), error);
}

DatasetStore::NameProblem DatasetStore::check_name(const std::string& name) {
    if (name.empty()) {
        return NameProblem::Empty;
    }
    if (name.size() > kMaxNameLength) {
        return NameProblem::TooLong;
    }
    const bool plain = std::ranges::all_of(name, [](char c) {
        return std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_';
    });
    if (!plain) {
        return NameProblem::BadCharacters;
    }
    std::string upper = name;
    std::ranges::transform(upper, upper.begin(), [](char c) { return static_cast<char>(std::toupper(c)); });
    return upper == "MNIST" ? NameProblem::Reserved : NameProblem::None;
}

}
