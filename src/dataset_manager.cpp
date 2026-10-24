#include "dataset_manager.h"

#include "model/normalize.h"

#include <algorithm>
#include <array>
#include <format>

namespace game {

namespace {

const std::string kMnistName = "MNIST";

bool is_blank(const Pixels& pixels) {
    return std::ranges::all_of(pixels, [](std::uint8_t value) { return value == 0; });
}

}

DatasetManager::DatasetManager(RecognizerViewModel& view, DatasetStore store, const Tr& tr, StatusLine& status)
    : view_(view)
    , store_(std::move(store))
    , tr_(tr)
    , status_(status)
    , samples_(std::make_shared<const LabeledImages>()) {
    view_.datasetPrev.bind_to<DatasetManager, &DatasetManager::dataset_prev>(*this);
    view_.datasetNext.bind_to<DatasetManager, &DatasetManager::dataset_next>(*this);
    view_.addLabelUp.bind_to<DatasetManager, &DatasetManager::label_up, &DatasetManager::can_label_up>(*this);
    view_.addLabelDown.bind_to<DatasetManager, &DatasetManager::label_down, &DatasetManager::can_label_down>(*this);
    view_.createDataset.bind_to<DatasetManager, &DatasetManager::create_dataset>(*this);
    view_.deleteDataset.bind_to<DatasetManager, &DatasetManager::delete_dataset, &DatasetManager::can_delete_dataset>(
            *this);
    view_.deleteSample.bind_to<DatasetManager, &DatasetManager::delete_sample, &DatasetManager::can_delete_sample>(
            *this);
    view_.pagePrev.bind_to<DatasetManager, &DatasetManager::page_prev, &DatasetManager::can_page_prev>(*this);
    view_.pageNext.bind_to<DatasetManager, &DatasetManager::page_next, &DatasetManager::can_page_next>(*this);

    refresh_list(kMnistName);
}

void DatasetManager::set_mnist(std::shared_ptr<const LabeledImages> train) {
    mnist_train_ = std::move(train);
    if (!custom_active()) {
        select_dataset(kMnist);
    }
}

bool DatasetManager::can_add_canvas_digit() const {
    return custom_active() && !is_blank(view_.canvasPaint.pixels());
}

bool DatasetManager::add_canvas_digit() {
    if (!custom_active()) {
        status_.say("status.mnist_readonly");
        return false;
    }
    auto updated = std::make_shared<LabeledImages>(*samples_);
    updated->add(normalize_digit(view_.canvasPaint.pixels()), static_cast<std::uint8_t>(label_));
    if (!commit(updated)) {
        return false;
    }
    // Show the new sample: it is the last one, so the last page.
    selected_sample_ = samples_->size() - 1;
    page_ = position_[*selected_sample_] / kPageSamples;
    rebuild_grid();
    show_texts();
    status_.say("status.sample_added", {{"digit", std::to_string(label_)}, {"name", active_name()},
                                                {"count", std::to_string(samples_->size())}});
    return true;
}

void DatasetManager::sort_samples() {
    const std::vector<std::uint8_t>& labels = samples_->labels;
    std::array<std::size_t, kNumClasses> next{};
    for (const std::uint8_t label : labels) {
        ++next[label];
    }
    std::size_t start = 0;
    for (std::size_t& slot : next) {
        const std::size_t count = slot;
        slot = start;
        start += count;
    }
    order_.assign(labels.size(), 0);
    position_.assign(labels.size(), 0);
    for (std::size_t index = 0; index < labels.size(); ++index) {
        const std::size_t position = next[labels[index]]++;
        order_[position] = index;
        position_[index] = position;
    }
}

std::size_t DatasetManager::page_count() const {
    return std::max<std::size_t>(1, (samples_->size() + kPageSamples - 1) / kPageSamples);
}

void DatasetManager::select_dataset(std::size_t index) {
    active_ = index;
    selected_sample_.reset();
    page_ = 0;
    if (index == kMnist) {
        samples_ = mnist_train_ ? mnist_train_ : std::make_shared<const LabeledImages>();
    } else {
        auto loaded = store_.load(names_[index]);
        if (!loaded) {
            status_.say_raw(loaded.error());
        }
        samples_ = std::make_shared<const LabeledImages>(loaded ? std::move(*loaded) : LabeledImages{});
    }
    sort_samples();
    update_labels();
    rebuild_grid();
    show_texts();
}

void DatasetManager::step_dataset(int direction) {
    const std::size_t count = names_.size();
    select_dataset((active_ + (direction > 0 ? 1 : count - 1)) % count);
}

void DatasetManager::label_up() {
    view_.addLabelText = std::to_string(++label_);
}

void DatasetManager::label_down() {
    view_.addLabelText = std::to_string(--label_);
}

void DatasetManager::create_dataset() {
    const std::string name = view_.newDatasetName.get();
    switch (DatasetStore::check_name(name)) {
    case DatasetStore::NameProblem::None: break;
    case DatasetStore::NameProblem::Empty: status_.say("name.empty"); return;
    case DatasetStore::NameProblem::TooLong:
        status_.say("name.too_long", {{"max", std::to_string(DatasetStore::kMaxNameLength)}});
        return;
    case DatasetStore::NameProblem::BadCharacters: status_.say("name.bad_chars"); return;
    case DatasetStore::NameProblem::Reserved: status_.say("name.reserved"); return;
    }
    if (std::ranges::find(names_, name) != names_.end()) {
        status_.say("status.dataset_exists", {{"name", name}});
        return;
    }
    if (const auto saved = store_.save(name, LabeledImages{}); !saved) {
        status_.say_raw(saved.error());
        return;
    }
    view_.newDatasetName = std::string{};
    refresh_list(name);
    status_.say("status.dataset_created", {{"name", name}});
}

void DatasetManager::delete_dataset() {
    const std::string name = active_name();
    if (!store_.remove(name)) {
        status_.say("status.dataset_delete_failed", {{"name", name}});
        return;
    }
    refresh_list(kMnistName);
    status_.say("status.dataset_deleted", {{"name", name}});
}

void DatasetManager::delete_sample() {
    const std::size_t index = *selected_sample_;
    auto updated = std::make_shared<LabeledImages>(*samples_);
    updated->remove(index);
    if (!commit(updated)) {
        return;
    }
    selected_sample_.reset();
    page_ = std::min(page_, page_count() - 1);
    rebuild_grid();
    show_texts();
    status_.say("status.sample_deleted", {{"n", std::to_string(index + 1)}, {"name", active_name()}});
}

bool DatasetManager::commit(std::shared_ptr<const LabeledImages> updated) {
    if (const auto saved = store_.save(active_name(), *updated); !saved) {
        status_.say("status.save_error", {{"detail", saved.error()}});
        return false;
    }
    samples_ = std::move(updated);
    sort_samples();
    show_texts();
    return true;
}

void DatasetManager::page_prev() {
    --page_;
    rebuild_grid();
    show_texts();
}

void DatasetManager::page_next() {
    ++page_;
    rebuild_grid();
    show_texts();
}

void DatasetManager::select_sample(std::size_t index) {
    const auto cell_of = [this](std::size_t sample) -> std::optional<GridCell*> {
        const std::size_t position = position_[sample];
        if (position < page_first() || position >= page_first() + kPageSamples) {
            return std::nullopt;
        }
        const std::size_t offset = position - page_first();
        return &rows_[offset / GridRow::kCells]->cell(offset % GridRow::kCells);
    };
    if (selected_sample_) {
        if (const auto cell = cell_of(*selected_sample_)) {
            (*cell)->set_selected(false);
        }
    }
    selected_sample_ = index;
    if (const auto cell = cell_of(index)) {
        (*cell)->set_selected(true);
    }
    show_texts();
}

void DatasetManager::refresh_list(const std::string& select_name) {
    names_ = {kMnistName};
    for (std::string& name : store_.list()) {
        names_.push_back(std::move(name));
    }

    items_.clear();
    for (std::size_t i = 0; i < names_.size(); ++i) {
        auto item = std::make_shared<ListItem>();
        item->select = [this, i] { select_dataset(i); };
        items_.push_back(std::move(item));
    }
    view_.datasets.set(items_);

    const auto found = std::ranges::find(names_, select_name);
    select_dataset(found == names_.end() ? kMnist : static_cast<std::size_t>(found - names_.begin()));
}

void DatasetManager::update_labels() {
    for (std::size_t i = 0; i < items_.size(); ++i) {
        items_[i]->label = (active_ == i ? "> " : "   ") + names_[i];
    }
}

void DatasetManager::rebuild_grid() {
    rows_.clear();
    const std::size_t end = std::min(samples_->size(), page_first() + kPageSamples);
    for (std::size_t first = page_first(); first < end; first += GridRow::kCells) {
        const std::span<const std::size_t> indices =
                std::span<const std::size_t>(order_).subspan(first, std::min(GridRow::kCells, end - first));
        auto row = std::make_shared<GridRow>(samples_, indices);
        for (std::size_t column = 0; column < indices.size(); ++column) {
            row->pick(column) = [this, index = indices[column]] { select_sample(index); };
        }
        rows_.push_back(std::move(row));
    }
    if (selected_sample_) {
        const std::size_t position = position_[*selected_sample_];
        if (position >= page_first() && position < end) {
            const std::size_t offset = position - page_first();
            rows_[offset / GridRow::kCells]->cell(offset % GridRow::kCells).set_selected(true);
        }
    }
    view_.gridRows.set(rows_);
}

void DatasetManager::show_texts() {
    const std::size_t count = samples_->size();
    const std::string total = std::to_string(count);
    view_.datasetNameText = tr_("ds.name", {{"name", active_name()}, {"count", total}});
    view_.datasetTitleText = custom_active() ? tr_("ds.title_custom", {{"name", active_name()}, {"count", total}})
                                             : tr_("ds.title_mnist", {{"count", total}});

    std::array<std::size_t, kNumClasses> per_digit{};
    for (const std::uint8_t label : samples_->labels) {
        ++per_digit[label];
    }
    std::string counts;
    for (std::size_t digit = 0; digit < kNumClasses; ++digit) {
        counts += std::format("{}: {}{}", digit, per_digit[digit], digit + 1 < kNumClasses ? ",  " : "");
    }
    view_.datasetCountsText = tr_("ds.counts", {{"list", counts}});
    view_.pageText = tr_("ds.page",
            {{"first", std::to_string(count == 0 ? 0 : page_first() + 1)},
                    {"last", std::to_string(std::min(count, page_first() + kPageSamples))},
                    {"total", total}});
    view_.addTargetText = custom_active() ? tr_("ds.add_target", {{"name", active_name()}})
                                          : tr_("ds.add_pick");
    if (selected_sample_ && *selected_sample_ < count) {
        view_.sampleInfoText = tr_("ds.sample",
                {{"n", std::to_string(*selected_sample_ + 1)},
                        {"digit", std::to_string(samples_->labels[*selected_sample_])}});
    } else {
        view_.sampleInfoText = tr_("ds.no_sample");
    }
}

}