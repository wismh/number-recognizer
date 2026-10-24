#include "model_library.h"

#include <algorithm>
#include <chrono>

namespace game {

ModelLibrary::ModelLibrary(RecognizerViewModel& view, ModelStore store)
    : view_(view)
    , store_(std::move(store)) {
    rebuild({});
}

void ModelLibrary::rebuild(const std::string& select_name) {
    names_ = store_.list();
    selected_.reset();
    if (const auto found = std::ranges::find(names_, select_name); found != names_.end()) {
        selected_ = static_cast<std::size_t>(found - names_.begin());
    } else if (!names_.empty()) {
        selected_ = 0;
    }

    items_.clear();
    for (std::size_t i = 0; i < names_.size(); ++i) {
        auto item = std::make_shared<ListItem>();
        item->select = [this, i] { select(i); };
        items_.push_back(std::move(item));
    }
    update_labels();
    view_.models.set(items_);
}

void ModelLibrary::refresh() {
    rebuild(selected_ ? names_[*selected_] : std::string{});
}

void ModelLibrary::start_load() {
    loading_name_ = names_[*selected_];
    load_ = std::async(std::launch::async, [path = store_.path_for(loading_name_)] { return Mlp::load(path); });
}

std::optional<ModelLibrary::Loaded> ModelLibrary::poll_load() {
    if (!load_.valid() || load_.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
        return std::nullopt;
    }
    return Loaded{.name = loading_name_, .model = load_.get()};
}

bool ModelLibrary::remove_selected() {
    if (!store_.remove(names_[*selected_])) {
        return false;
    }
    // The next model in the list takes the selection.
    const std::size_t next = *selected_;
    rebuild(next + 1 < names_.size() ? names_[next + 1] : std::string{});
    return true;
}

std::future<std::expected<void, std::string>> ModelLibrary::save(std::shared_ptr<const Mlp> model,
        const std::string& file_name) const {
    return std::async(std::launch::async,
            [model = std::move(model), path = store_.path_for(file_name)] { return model->save(path); });
}

void ModelLibrary::select(std::size_t index) {
    selected_ = index;
    update_labels();
}

void ModelLibrary::update_labels() {
    for (std::size_t i = 0; i < items_.size(); ++i) {
        items_[i]->label = (selected_ == i ? "> " : "   ") + names_[i];
    }
}

}
