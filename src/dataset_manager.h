#pragma once

#include "loc/tr.h"
#include "model/dataset_store.h"
#include "model/labeled_images.h"
#include "status_line.h"
#include "view/recognizer_view_model.h"

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace game {

// The datasets of the app and the one in use. MNIST is always first and cannot be changed; the rest are the user's own
// files, which can be created, deleted, and have samples added and removed. The one in use is what training runs on,
// what "Add" adds to, and what the Datasets tab shows as a grid.
class DatasetManager {
public:
    static constexpr std::size_t kMnist = 0;
    // The grid shows one page of this many samples (a multiple of GridRow::kCells). The UI's cost of a list grows
    // faster than its length: 500 rows open at once, 2 000 take a second, 5 000 take seconds. 6 000 samples are 500 rows.
    static constexpr std::size_t kPageSamples = 6000;

    DatasetManager(RecognizerViewModel& view, DatasetStore store, const Tr& tr, StatusLine& status);

    DatasetManager(const DatasetManager&) = delete;
    DatasetManager& operator=(const DatasetManager&) = delete;

    // Writes the texts of the Datasets tab again after a change of language.
    void apply_locale() { show_texts(); }

    // Gives MNIST its 60 000 training samples once they are read. Until then MNIST is an empty dataset.
    void set_mnist(std::shared_ptr<const LabeledImages> train);

    // A snapshot of the dataset in use: later changes to the dataset do not touch it.
    [[nodiscard]] std::shared_ptr<const LabeledImages> active_samples() const { return samples_; }
    [[nodiscard]] const std::string& active_name() const { return names_[active_]; }

    // Normalizes the digit on the drawing canvas like MNIST digits and adds it to the dataset in use with the chosen
    // label. Returns false when that failed (the status line says why).
    bool add_canvas_digit();
    [[nodiscard]] bool can_add_canvas_digit() const;

private:
    using Rows = std::vector<std::shared_ptr<GridRow>>;

    [[nodiscard]] bool custom_active() const { return active_ != kMnist; }
    [[nodiscard]] std::size_t page_count() const;
    [[nodiscard]] std::size_t page_first() const { return page_ * kPageSamples; }
    // Lists the samples by digit (0s first, each digit in the order it was added): order_[position] is the index of
    // the sample shown at that position of the grid, position_[index] the other way round.
    void sort_samples();

    void select_dataset(std::size_t index);
    void step_dataset(int direction);
    void dataset_prev() { step_dataset(-1); }
    void dataset_next() { step_dataset(1); }
    void label_up();
    void label_down();
    [[nodiscard]] bool can_label_up() const { return label_ < kNumClasses - 1; }
    [[nodiscard]] bool can_label_down() const { return label_ > 0; }
    void create_dataset();
    void delete_dataset();
    void delete_sample();
    [[nodiscard]] bool can_delete_dataset() const { return custom_active(); }
    [[nodiscard]] bool can_delete_sample() const { return custom_active() && selected_sample_.has_value(); }
    void page_prev();
    void page_next();
    [[nodiscard]] bool can_page_prev() const { return page_ > 0; }
    [[nodiscard]] bool can_page_next() const { return page_ + 1 < page_count(); }
    void select_sample(std::size_t index);

    void refresh_list(const std::string& select_name);
    void update_labels();
    void rebuild_grid();
    void show_texts();
    // Replaces the samples of the custom dataset in use with `updated`, saving them first.
    bool commit(std::shared_ptr<const LabeledImages> updated);

    RecognizerViewModel& view_;
    DatasetStore store_;
    const Tr& tr_;
    StatusLine& status_;
    std::shared_ptr<const LabeledImages> mnist_train_;

    std::vector<std::string> names_;  // names_[kMnist] is "MNIST"
    std::vector<std::shared_ptr<ListItem>> items_;
    std::size_t active_ = kMnist;
    std::shared_ptr<const LabeledImages> samples_;
    std::vector<std::size_t> order_;
    std::vector<std::size_t> position_;
    std::size_t page_ = 0;  // a page is kPageSamples positions of the grid
    Rows rows_;             // the rows of the page shown
    std::optional<std::size_t> selected_sample_;  // index in the dataset, not its position in the grid
    std::size_t label_ = 0;
};

}
