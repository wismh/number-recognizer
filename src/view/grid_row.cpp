#include "view/grid_row.h"

#include <engine/builtin_ids.h>

#include <format>

namespace game {

void GridCell::paint(engine::ui::IDrawList& list, const engine::render::Rect& content) {
    if (index_ == kNone) {
        return;
    }

    constexpr int kSide = 28;
    const float pixel = content.w / kSide;
    list.fill_rect(content, {0.0f, 0.0f, 0.0f, 1.0f});

    const std::span<const std::uint8_t> image = samples_->image(index_);
    for (int y = 0; y < kSide; ++y) {
        for (int x = 0; x < kSide; ++x) {
            const std::uint8_t raw = image[static_cast<std::size_t>(y * kSide + x)];
            if (raw > 0) {
                const float value = static_cast<float>(raw) / 255.0f;
                list.fill_rect({content.x + x * pixel, content.y + y * pixel, pixel, pixel},
                        {value, value, value, 1.0f});
            }
        }
    }

    list.set_font(engine::builtin::font_ui, 11.0f);
    list.text(std::format("{}", samples_->labels[index_]), {content.x + 3.0f, content.y + 1.0f},
            {0.35f, 0.85f, 0.45f, 1.0f});
    if (selected_) {
        list.stroke_rect(content, {1.0f, 0.78f, 0.2f, 1.0f}, 2.0f);
    }
}

void GridCell::show(const LabeledImages* samples, std::size_t index) {
    samples_ = samples;
    index_ = index;
    ++version_;
}

void GridCell::set_selected(bool selected) {
    if (selected_ != selected) {
        selected_ = selected;
        ++version_;
    }
}

namespace {

// "cell0".."cell11" and "pick0".."pick11", hashed once: a big dataset builds thousands of rows.
struct ColumnIds {
    std::array<engine::ui::BindingId, GridRow::kCells> cells;
    std::array<engine::ui::BindingId, GridRow::kCells> picks;

    ColumnIds() {
        for (std::size_t column = 0; column < GridRow::kCells; ++column) {
            cells[column] = engine::ui::intern(std::format("cell{}", column));
            picks[column] = engine::ui::intern(std::format("pick{}", column));
        }
    }
};

const ColumnIds& column_ids() {
    static const ColumnIds ids;
    return ids;
}

}

GridRow::GridRow(std::shared_ptr<const LabeledImages> samples, std::span<const std::size_t> indices)
    : samples_(std::move(samples)) {
    for (std::size_t column = 0; column < kCells; ++column) {
        const bool used = column < indices.size();
        cells_[column].show(samples_.get(), used ? indices[column] : GridCell::kNone);
        picks_[column].set_can_execute(used);

        paint(column_ids().cells[column], cells_[column]);
        command(column_ids().picks[column], picks_[column]);
    }
}

}
