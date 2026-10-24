#pragma once

#include "model/mlp.h"
#include "model/model_store.h"
#include "view/recognizer_view_model.h"

#include <expected>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace game {

// The saved models: the list on the Train tab with the selection in it, and the files behind it. Reading and writing a
// file happens on another thread; the caller polls for the result once per frame.
class ModelLibrary {
public:
    struct Loaded {
        std::string name;
        std::expected<Mlp, std::string> model;
    };

    ModelLibrary(RecognizerViewModel& view, ModelStore store);

    ModelLibrary(const ModelLibrary&) = delete;
    ModelLibrary& operator=(const ModelLibrary&) = delete;

    [[nodiscard]] bool has_selection() const { return selected_.has_value(); }
    // The file the list has selected; only when has_selection().
    [[nodiscard]] const std::string& selected_name() const { return names_[*selected_]; }
    [[nodiscard]] bool loading() const { return load_.valid(); }
    [[nodiscard]] bool can_save() const { return store_.available(); }

    // Reads the folder again and selects `select_name`, or the first model when there is none of that name.
    void rebuild(const std::string& select_name);
    // The same, keeping the selection.
    void refresh();
    // Starts reading the selected file.
    void start_load();
    // The result of the read once it is done.
    [[nodiscard]] std::optional<Loaded> poll_load();
    // Deletes the selected file and selects the next one. False when the file could not be deleted.
    bool remove_selected();
    // Writes `model` as `file_name` on another thread.
    [[nodiscard]] std::future<std::expected<void, std::string>> save(std::shared_ptr<const Mlp> model,
            const std::string& file_name) const;

private:
    void select(std::size_t index);
    void update_labels();

    RecognizerViewModel& view_;
    ModelStore store_;
    std::vector<std::string> names_;
    std::vector<std::shared_ptr<ListItem>> items_;
    std::optional<std::size_t> selected_;
    std::future<std::expected<Mlp, std::string>> load_;
    std::string loading_name_;
};

}
