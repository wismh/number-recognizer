#include "model/model_store.h"

#include <algorithm>
#include <charconv>
#include <cctype>
#include <chrono>
#include <format>

namespace game {

ModelStore::ModelStore(std::filesystem::path directory)
    : directory_(std::move(directory)) {}

std::vector<std::string> ModelStore::list() const {
    std::vector<std::string> names;
    if (!available()) {
        return names;
    }
    std::error_code error;
    for (const auto& entry : std::filesystem::directory_iterator(directory_, error)) {
        if (entry.is_regular_file(error) && entry.path().extension() == ".bin") {
            names.push_back(entry.path().filename().string());
        }
    }
    std::ranges::sort(names);
    return names;
}

std::filesystem::path ModelStore::path_for(const std::string& name) const {
    return directory_ / std::filesystem::path(name).filename();
}

bool ModelStore::remove(const std::string& name) const {
    std::error_code error;
    return available() && std::filesystem::remove(path_for(name), error);
}

unsigned ModelStore::epochs_in_name(const std::string& name) {
    const std::size_t end = name.find("ep_");
    if (end == std::string::npos) {
        return 0;
    }
    std::size_t begin = end;
    while (begin > 0 && std::isdigit(static_cast<unsigned char>(name[begin - 1]))) {
        --begin;
    }
    unsigned epochs = 0;
    const auto [last, error] = std::from_chars(name.data() + begin, name.data() + end, epochs);
    return error == std::errc{} && last == name.data() + end ? epochs : 0;
}

std::string ModelStore::new_file_name(const std::string& architecture, unsigned epochs) {
    const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch())
                                 .count();
    return std::format("model_{}_{}ep_{}.bin", architecture, epochs, seconds);
}

}
