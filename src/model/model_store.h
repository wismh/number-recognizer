#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace game {

// Saved models are `*.bin` files in one directory. An empty directory means there is no storage: the list is empty
// and nothing can be saved or removed.
class ModelStore {
public:
    explicit ModelStore(std::filesystem::path directory);

    // Sorted file names.
    [[nodiscard]] std::vector<std::string> list() const;
    [[nodiscard]] std::filesystem::path path_for(const std::string& name) const;
    // Deletes the file; false when it could not be deleted.
    bool remove(const std::string& name) const;

    // `model_<architecture>_<epochs>ep_<unix time>.bin`; `architecture` is a short label such as "2x64".
    [[nodiscard]] static std::string new_file_name(const std::string& architecture, unsigned epochs);

    [[nodiscard]] bool available() const { return !directory_.empty(); }

private:
    std::filesystem::path directory_;
};

}
