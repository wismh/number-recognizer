#include "help/help_figures.h"

#include <engine/builtin_ids.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <numbers>
#include <string>

namespace game {

namespace {

using engine::render::Rect;
using engine::ui::IDrawList;

constexpr glm::vec4 kInk{0.20f, 0.25f, 0.33f, 1.0f};
constexpr glm::vec4 kMuted{0.58f, 0.64f, 0.72f, 1.0f};
constexpr glm::vec4 kGrid{0.89f, 0.91f, 0.94f, 1.0f};
constexpr glm::vec4 kBlue{0.15f, 0.39f, 0.92f, 1.0f};
constexpr glm::vec4 kGreen{0.09f, 0.64f, 0.29f, 1.0f};
constexpr glm::vec4 kRed{0.86f, 0.15f, 0.15f, 1.0f};
constexpr glm::vec4 kPaper{1.0f, 1.0f, 1.0f, 1.0f};
constexpr glm::vec4 kBlack{0.0f, 0.0f, 0.0f, 1.0f};
constexpr float kPi = std::numbers::pi_v<float>;

// A small drawing board: text, dots and polylines in the area of one figure.
class Board {
public:
    Board(IDrawList& list, const Rect& area, const Tr& tr)
        : list_(list)
        , area_(area)
        , tr_(tr) {}

    [[nodiscard]] const Rect& area() const { return area_; }
    [[nodiscard]] std::string tr(const char* key) const { return tr_(key); }

    void line(glm::vec2 a, glm::vec2 b, glm::vec4 color, float width = 1.5f) {
        list_.line(abs(a), abs(b), color, width);
    }

    void dashed(glm::vec2 a, glm::vec2 b, glm::vec4 color, float width = 1.0f) {
        const glm::vec2 from = abs(a);
        const glm::vec2 to = abs(b);
        const float length = glm::length(to - from);
        const int pieces = std::max(1, static_cast<int>(length / 8.0f));
        for (int i = 0; i < pieces; ++i) {
            const float t0 = static_cast<float>(i) / static_cast<float>(pieces);
            const float t1 = (static_cast<float>(i) + 0.55f) / static_cast<float>(pieces);
            list_.line(from + (to - from) * t0, from + (to - from) * t1, color, width);
        }
    }

    void polyline(const std::vector<glm::vec2>& points, glm::vec4 color, float width = 2.0f) {
        for (std::size_t i = 1; i < points.size(); ++i) {
            list_.line(abs(points[i - 1]), abs(points[i]), color, width);
        }
    }

    void dot(glm::vec2 centre, float radius, glm::vec4 color) {
        const glm::vec2 c = abs(centre);
        list_.fill_rect({c.x - radius, c.y - radius, radius * 2.0f, radius * 2.0f}, color, radius);
    }

    void box(const Rect& rect, glm::vec4 fill, glm::vec4 border, float width = 1.0f) {
        const Rect r{area_.x + rect.x, area_.y + rect.y, rect.w, rect.h};
        list_.fill_rect(r, fill);
        list_.stroke_rect(r, border, width);
    }

    void fill(const Rect& rect, glm::vec4 color, float radius = 0.0f) {
        list_.fill_rect({area_.x + rect.x, area_.y + rect.y, rect.w, rect.h}, color, radius);
    }

    // `at` is the top-left of the text; `align` 0 left, 0.5 centred, 1 right of that point (width is estimated).
    void text(const std::string& value, glm::vec2 at, glm::vec4 color, float size = 13.0f, float align = 0.0f) {
        list_.set_font(engine::builtin::font_ui, size);
        const float width = static_cast<float>(value.size()) * size * 0.52f;
        list_.text(value, abs({at.x - width * align, at.y}), color);
    }

private:
    [[nodiscard]] glm::vec2 abs(glm::vec2 p) const { return {area_.x + p.x, area_.y + p.y}; }

    IDrawList& list_;
    Rect area_;
    const Tr& tr_;
};

// A plot with its own coordinates: world (x, y) maps into a margin-inset part of the board, y pointing up.
struct Plot {
    Board& board;
    Rect frame;  // in board pixels
    float x0, x1, y0, y1;

    [[nodiscard]] glm::vec2 at(float x, float y) const {
        return {frame.x + (x - x0) / (x1 - x0) * frame.w, frame.y + frame.h - (y - y0) / (y1 - y0) * frame.h};
    }

    void grid(float step_x, float step_y) const {
        for (float x = std::ceil(x0 / step_x) * step_x; x <= x1 + 1e-4f; x += step_x) {
            board.line(at(x, y0), at(x, y1), kGrid, 1.0f);
        }
        for (float y = std::ceil(y0 / step_y) * step_y; y <= y1 + 1e-4f; y += step_y) {
            board.line(at(x0, y), at(x1, y), kGrid, 1.0f);
        }
    }

    void axes() const {
        const float ax = std::clamp(0.0f, x0, x1);
        const float ay = std::clamp(0.0f, y0, y1);
        board.line(at(x0, ay), at(x1, ay), kMuted, 1.5f);
        board.line(at(ax, y0), at(ax, y1), kMuted, 1.5f);
    }

    [[nodiscard]] std::vector<glm::vec2> curve(float from, float to, int steps, float (*f)(float)) const {
        std::vector<glm::vec2> points;
        for (int i = 0; i <= steps; ++i) {
            const float x = from + (to - from) * static_cast<float>(i) / static_cast<float>(steps);
            points.push_back(at(x, f(x)));
        }
        return points;
    }
};

// A stylized digit 7 drawn with strokes inside `frame` (board pixels), for the augmentation and normalization pictures.
void seven(Board& board, const Rect& frame, glm::vec4 color, float width) {
    const auto p = [&](float u, float v) { return glm::vec2{frame.x + u * frame.w, frame.y + v * frame.h}; };
    board.line(p(0.0f, 0.0f), p(1.0f, 0.0f), color, width);
    board.line(p(1.0f, 0.0f), p(0.35f, 1.0f), color, width);
}

constexpr std::array<float, 10> kScores{-1.2f, 3.1f, 0.4f, -0.5f, 1.0f, 0.2f, -2.0f, 0.9f, 1.6f, -0.3f};

// Ten bars around a zero line; `values` may go below it. The tallest one is blue.
void bars(Board& board, const Rect& frame, const std::array<float, 10>& values, float low, float high,
        bool percent_labels) {
    const Plot plot{board, frame, 0.0f, 10.0f, low, high};
    board.line(plot.at(0.0f, 0.0f), plot.at(10.0f, 0.0f), kMuted, 1.5f);
    const auto best = std::ranges::max_element(values) - values.begin();
    for (std::size_t k = 0; k < values.size(); ++k) {
        const glm::vec2 top = plot.at(static_cast<float>(k) + 0.2f, std::max(values[k], 0.0f));
        const glm::vec2 bottom = plot.at(static_cast<float>(k) + 0.8f, std::min(values[k], 0.0f));
        board.fill({top.x, top.y, bottom.x - top.x, std::max(1.0f, bottom.y - top.y)},
                static_cast<long>(k) == best ? kBlue : kMuted, 2.0f);
        board.text(std::to_string(k), {top.x + (bottom.x - top.x) * 0.5f, frame.y + frame.h + 4.0f}, kInk, 12.0f, 0.5f);
        if (percent_labels && values[k] > 0.04f) {
            board.text(std::format("{:.0f}%", values[k] * 100.0f), {top.x + (bottom.x - top.x) * 0.5f, top.y - 15.0f},
                    kInk, 11.0f, 0.5f);
        }
    }
}

void line_figure(Board& board, const LineParams& line) {
    const Rect area = board.area();
    constexpr float kX0 = -1.0f;
    constexpr float kX1 = 7.0f;
    constexpr float kY0 = -1.0f;
    constexpr float kY1 = 5.0f;
    const Plot plot{board, {34.0f, 14.0f, area.w - 60.0f, area.h - 48.0f}, kX0, kX1, kY0, kY1};
    plot.grid(1.0f, 1.0f);
    plot.axes();

    // The part of the line inside the plot: the x range where y = w x + b stays between kY0 and kY1.
    float from = kX0;
    float to = kX1;
    if (std::abs(line.w) > 1e-4f) {
        const float at_low = (kY0 - line.b) / line.w;
        const float at_high = (kY1 - line.b) / line.w;
        from = std::max(from, std::min(at_low, at_high));
        to = std::min(to, std::max(at_low, at_high));
    }
    if (from < to) {
        board.polyline({plot.at(from, line.w * from + line.b), plot.at(to, line.w * to + line.b)}, kBlue, 3.0f);
    }
    board.dot(plot.at(0.0f, line.b), 5.0f, kRed);
    board.text("b", plot.at(0.0f, line.b) + glm::vec2{-18.0f, -8.0f}, kRed, 15.0f);
    board.text(std::format("y = {:.1f} x {} {:.1f}", line.w, line.b < 0.0f ? '-' : '+', std::abs(line.b)),
            plot.at(2.6f, 4.7f), kBlue, 15.0f);
    board.text("x", plot.at(7.0f, 0.0f) + glm::vec2{-2.0f, 6.0f}, kInk, 13.0f);
    board.text("y", plot.at(0.0f, 5.0f) + glm::vec2{8.0f, -2.0f}, kInk, 13.0f);
}

void pixels_figure(Board& board) {
    // A coarse 8 x 8 picture of a 7, with brightness values written in the cells.
    constexpr int kCells = 8;
    constexpr float kCell = 22.0f;
    constexpr float kLeft = 24.0f;
    constexpr float kTop = 14.0f;
    const std::array<std::array<float, kCells>, kCells> image{{
            {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
            {0.0f, 0.9f, 1.0f, 1.0f, 1.0f, 1.0f, 0.8f, 0.0f},
            {0.0f, 0.0f, 0.0f, 0.0f, 0.3f, 0.9f, 0.6f, 0.0f},
            {0.0f, 0.0f, 0.0f, 0.2f, 0.9f, 0.7f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f, 0.7f, 0.9f, 0.1f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.4f, 1.0f, 0.3f, 0.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.8f, 0.8f, 0.0f, 0.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
    }};
    for (int r = 0; r < kCells; ++r) {
        for (int c = 0; c < kCells; ++c) {
            const float v = image[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)];
            const Rect cell{kLeft + c * kCell, kTop + r * kCell, kCell - 1.0f, kCell - 1.0f};
            board.fill(cell, {v, v, v, 1.0f});
            board.text(v == 0.0f ? "0" : (v == 1.0f ? "1" : std::format("{:.1f}", v).substr(1)),
                    {cell.x + cell.w * 0.5f, cell.y + 5.0f}, v > 0.5f ? kBlack : glm::vec4{0.75f, 0.78f, 0.84f, 1.0f},
                    10.0f, 0.5f);
        }
    }
    board.text(board.tr("fig.pixel_values"), {kLeft + kCells * kCell * 0.5f, kTop + kCells * kCell + 6.0f}, kInk, 12.0f,
            0.5f);

    // The same numbers read row by row as one list.
    const float list_x = kLeft + kCells * kCell + 90.0f;
    board.line({kLeft + kCells * kCell + 16.0f, kTop + kCells * kCell * 0.5f},
            {kLeft + kCells * kCell + 62.0f, kTop + kCells * kCell * 0.5f}, kMuted, 2.0f);
    board.line({kLeft + kCells * kCell + 52.0f, kTop + kCells * kCell * 0.5f - 7.0f},
            {kLeft + kCells * kCell + 62.0f, kTop + kCells * kCell * 0.5f}, kMuted, 2.0f);
    board.line({kLeft + kCells * kCell + 52.0f, kTop + kCells * kCell * 0.5f + 7.0f},
            {kLeft + kCells * kCell + 62.0f, kTop + kCells * kCell * 0.5f}, kMuted, 2.0f);
    // The first numbers of the list, then the last one.
    const std::array<const char*, 6> head{"0", "0", "0", "0", "0", "0"};
    for (std::size_t i = 0; i < head.size(); ++i) {
        const Rect cell{list_x, kTop + static_cast<float>(i) * 22.0f, 46.0f, 20.0f};
        board.box(cell, kPaper, kMuted);
        board.text(head[i], {cell.x + cell.w * 0.5f, cell.y + 3.0f}, kInk, 13.0f, 0.5f);
    }
    board.text("...", {list_x + 23.0f, kTop + 6 * 22.0f - 2.0f}, kInk, 16.0f, 0.5f);
    const Rect last{list_x, kTop + 6 * 22.0f + 22.0f, 46.0f, 20.0f};
    board.box(last, kPaper, kMuted);
    board.text("0", {last.x + last.w * 0.5f, last.y + 3.0f}, kInk, 13.0f, 0.5f);
    board.text(board.tr("fig.vector"), {list_x + 23.0f, kTop + kCells * kCell + 6.0f}, kInk, 12.0f, 0.5f);
}
void scores_figure(Board& board) {
    const Rect area = board.area();
    bars(board, {30.0f, 28.0f, area.w - 60.0f, area.h - 70.0f}, kScores, -2.5f, 3.5f, false);
    board.text(board.tr("fig.scores"), {area.w * 0.5f, 4.0f}, kInk, 13.0f, 0.5f);
}

std::array<float, 10> softmax_of(const std::array<float, 10>& scores) {
    std::array<float, 10> result{};
    const float max = *std::ranges::max_element(scores);
    float sum = 0.0f;
    for (std::size_t k = 0; k < scores.size(); ++k) {
        result[k] = std::exp(scores[k] - max);
        sum += result[k];
    }
    for (float& value : result) {
        value /= sum;
    }
    return result;
}

void softmax_figure(Board& board) {
    const Rect area = board.area();
    const float half = (area.w - 70.0f) * 0.5f;
    bars(board, {20.0f, 30.0f, half, area.h - 74.0f}, kScores, -2.5f, 3.5f, false);
    bars(board, {50.0f + half, 30.0f, half, area.h - 74.0f}, softmax_of(kScores), 0.0f, 0.8f, true);
    board.text(board.tr("fig.scores"), {20.0f + half * 0.5f, 6.0f}, kInk, 13.0f, 0.5f);
    board.text(board.tr("fig.probabilities"), {50.0f + half * 1.5f, 6.0f}, kInk, 13.0f, 0.5f);
    board.line({26.0f + half, area.h * 0.5f}, {44.0f + half, area.h * 0.5f}, kMuted, 2.0f);
    board.line({36.0f + half, area.h * 0.5f - 6.0f}, {44.0f + half, area.h * 0.5f}, kMuted, 2.0f);
    board.line({36.0f + half, area.h * 0.5f + 6.0f}, {44.0f + half, area.h * 0.5f}, kMuted, 2.0f);
}

void loss_figure(Board& board) {
    const Rect area = board.area();
    const Plot plot{board, {40.0f, 14.0f, area.w - 70.0f, area.h - 56.0f}, 0.0f, 1.0f, 0.0f, 3.6f};
    plot.grid(0.25f, 0.5f);
    plot.axes();
    std::vector<glm::vec2> points;
    for (int i = 0; i <= 80; ++i) {
        const float p = 0.027f + (1.0f - 0.027f) * static_cast<float>(i) / 80.0f;
        points.push_back(plot.at(p, -std::log(p)));
    }
    board.polyline(points, kBlue, 3.0f);
    for (const float p : {0.1f, 0.5f, 1.0f}) {
        board.dot(plot.at(p, -std::log(p)), 4.5f, kRed);
        board.text(std::format("{:.2f}", -std::log(p)), plot.at(p, -std::log(p)) + glm::vec2{8.0f, -16.0f}, kRed, 12.0f);
    }
    board.text(board.tr("fig.prob_correct"), {40.0f + (area.w - 70.0f) * 0.5f, area.h - 22.0f}, kInk, 12.0f, 0.5f);
    board.text(board.tr("fig.loss"), {44.0f, 2.0f}, kInk, 12.0f);
}

void descent_figure(Board& board) {
    const Rect area = board.area();
    const Plot plot{board, {40.0f, 14.0f, area.w - 70.0f, area.h - 56.0f}, 0.0f, 6.0f, 0.0f, 9.5f};
    plot.axes();
    board.polyline(plot.curve(0.0f, 6.0f, 60, [](float w) { return (w - 3.0f) * (w - 3.0f); }), kMuted, 3.0f);
    float w = 0.4f;
    for (int step = 0; step < 6; ++step) {
        const float next = w - 0.3f * 2.0f * (w - 3.0f);
        const glm::vec2 a = plot.at(w, (w - 3.0f) * (w - 3.0f));
        const glm::vec2 b = plot.at(next, (next - 3.0f) * (next - 3.0f));
        board.line(a, b, kBlue, 2.0f);
        board.dot(a, 5.0f, step == 0 ? kRed : kBlue);
        w = next;
    }
    board.dot(plot.at(w, (w - 3.0f) * (w - 3.0f)), 5.0f, kGreen);
    board.text(board.tr("fig.weight"), {40.0f + (area.w - 70.0f) * 0.5f, area.h - 22.0f}, kInk, 12.0f, 0.5f);
    board.text(board.tr("fig.loss"), {44.0f, 2.0f}, kInk, 12.0f);
}

void network_figure(Board& board) {
    const Rect area = board.area();
    constexpr std::array<int, 3> counts{8, 6, 10};
    std::array<float, 3> columns{area.w * 0.16f, area.w * 0.5f, area.w * 0.84f};
    const auto node = [&](int column, int i) {
        const int n = counts[static_cast<std::size_t>(column)];
        const float top = 22.0f;
        const float height = area.h - 70.0f;
        return glm::vec2{columns[static_cast<std::size_t>(column)],
                top + height * (static_cast<float>(i) + 0.5f) / static_cast<float>(n)};
    };
    for (int column = 0; column < 2; ++column) {
        for (int a = 0; a < counts[static_cast<std::size_t>(column)]; ++a) {
            for (int b = 0; b < counts[static_cast<std::size_t>(column) + 1]; ++b) {
                board.line(node(column, a), node(column + 1, b), {0.65f, 0.70f, 0.78f, 0.55f}, 1.0f);
            }
        }
    }
    for (int column = 0; column < 3; ++column) {
        for (int i = 0; i < counts[static_cast<std::size_t>(column)]; ++i) {
            board.dot(node(column, i), 7.0f, column == 1 ? kBlue : (column == 0 ? kInk : kGreen));
        }
    }
    board.text("784", {columns[0], 2.0f}, kInk, 12.0f, 0.5f);
    board.text("...", {columns[0], area.h - 66.0f}, kInk, 14.0f, 0.5f);
    board.text(board.tr("fig.input"), {columns[0], area.h - 40.0f}, kInk, 12.0f, 0.5f);
    board.text(board.tr("fig.hidden"), {columns[1], area.h - 40.0f}, kInk, 12.0f, 0.5f);
    board.text(board.tr("fig.scores"), {columns[2], area.h - 40.0f}, kInk, 12.0f, 0.5f);
    board.text("h", {columns[1], 2.0f}, kBlue, 12.0f, 0.5f);
    board.text("s, 0..9", {columns[2], 2.0f}, kGreen, 12.0f, 0.5f);
}

void relu_figure(Board& board) {
    const Rect area = board.area();
    const Plot plot{board, {30.0f, 12.0f, area.w - 60.0f, area.h - 34.0f}, -3.0f, 3.0f, -1.0f, 3.0f};
    plot.grid(1.0f, 1.0f);
    plot.axes();
    board.dashed(plot.at(-1.0f, -1.0f), plot.at(3.0f, 3.0f), kMuted, 1.5f);
    board.polyline({plot.at(-3.0f, 0.0f), plot.at(0.0f, 0.0f), plot.at(3.0f, 3.0f)}, kBlue, 3.5f);
    board.text("max(0, z)", plot.at(1.2f, 2.6f) + glm::vec2{-70.0f, -4.0f}, kBlue, 14.0f);
    board.text("z", plot.at(3.0f, 0.0f) + glm::vec2{-2.0f, 6.0f}, kInk, 13.0f);
}

void overfit_figure(Board& board) {
    const Rect area = board.area();
    const Plot plot{board, {40.0f, 14.0f, area.w - 70.0f, area.h - 56.0f}, 0.0f, 10.0f, 0.0f, 2.5f};
    plot.grid(2.0f, 0.5f);
    plot.axes();
    board.polyline(plot.curve(0.0f, 10.0f, 50, [](float t) { return 0.12f + 2.1f * std::exp(-0.55f * t); }), kBlue, 3.0f);
    board.polyline(plot.curve(0.0f, 10.0f, 50,
                           [](float t) { return 0.55f + 1.7f * std::exp(-0.7f * t) + 0.012f * t * t; }),
            kRed, 3.0f);
    board.dashed(plot.at(4.6f, 0.0f), plot.at(4.6f, 2.5f), kMuted, 1.5f);
    board.text(board.tr("fig.train"), plot.at(6.4f, 0.12f) + glm::vec2{0.0f, 10.0f}, kBlue, 12.0f);
    board.text(board.tr("fig.test"), plot.at(7.4f, 1.0f) + glm::vec2{0.0f, -22.0f}, kRed, 12.0f);
    board.text(board.tr("fig.epochs"), {40.0f + (area.w - 70.0f) * 0.5f, area.h - 22.0f}, kInk, 12.0f, 0.5f);
    board.text(board.tr("fig.loss"), {44.0f, 2.0f}, kInk, 12.0f);
}

void augment_figure(Board& board) {
    const Rect area = board.area();
    const float frame = std::min(120.0f, area.h - 56.0f);
    const std::array<float, 3> lefts{area.w * 0.2f - frame * 0.5f, area.w * 0.5f - frame * 0.5f,
            area.w * 0.8f - frame * 0.5f};
    const float top = 14.0f;
    const std::array<const char*, 3> names{"fig.original", "fig.shifted", "fig.scaled"};
    // The digit as a box inside the frame: centred, moved, bigger.
    const std::array<Rect, 3> digit{{
            {0.30f, 0.20f, 0.40f, 0.60f},
            {0.50f, 0.28f, 0.40f, 0.60f},
            {0.10f, 0.08f, 0.62f, 0.84f},
    }};
    for (std::size_t i = 0; i < 3; ++i) {
        board.box({lefts[i], top, frame, frame}, kBlack, kMuted, 2.0f);
        const Rect d = digit[i];
        seven(board, {lefts[i] + d.x * frame, top + d.y * frame, d.w * frame, d.h * frame}, {0.95f, 0.95f, 0.95f, 1.0f},
                3.0f);
        board.text(board.tr(names[i]), {lefts[i] + frame * 0.5f, top + frame + 8.0f}, kInk, 12.0f, 0.5f);
    }
}

void normalize_figure(Board& board) {
    const Rect area = board.area();
    const float frame = std::min(140.0f, area.h - 56.0f);
    const float left_a = area.w * 0.25f - frame * 0.5f;
    const float left_b = area.w * 0.75f - frame * 0.5f;
    const float top = 14.0f;

    // As drawn: big and off centre.
    board.box({left_a, top, frame, frame}, kBlack, kMuted, 2.0f);
    seven(board, {left_a + frame * 0.38f, top + frame * 0.06f, frame * 0.56f, frame * 0.84f}, {0.95f, 0.95f, 0.95f, 1.0f},
            3.0f);

    // Normalized: the 20 x 20 box, the centre of mass at the middle.
    board.box({left_b, top, frame, frame}, kBlack, kMuted, 2.0f);
    const float inner = frame * 20.0f / 28.0f;
    const float inner_left = left_b + (frame - inner) * 0.5f;
    const float inner_top = top + (frame - inner) * 0.5f;
    board.dashed({inner_left, inner_top}, {inner_left + inner, inner_top}, kGreen, 1.5f);
    board.dashed({inner_left + inner, inner_top}, {inner_left + inner, inner_top + inner}, kGreen, 1.5f);
    board.dashed({inner_left + inner, inner_top + inner}, {inner_left, inner_top + inner}, kGreen, 1.5f);
    board.dashed({inner_left, inner_top + inner}, {inner_left, inner_top}, kGreen, 1.5f);
    seven(board, {inner_left + inner * 0.15f, inner_top, inner * 0.7f, inner}, {0.95f, 0.95f, 0.95f, 1.0f}, 3.0f);
    const glm::vec2 centre{left_b + frame * 0.5f, top + frame * 0.5f};
    board.line(centre + glm::vec2{-7.0f, 0.0f}, centre + glm::vec2{7.0f, 0.0f}, kRed, 2.0f);
    board.line(centre + glm::vec2{0.0f, -7.0f}, centre + glm::vec2{0.0f, 7.0f}, kRed, 2.0f);
    board.text(board.tr("fig.box"), {inner_left + inner * 0.5f, top + frame + 8.0f}, kGreen, 12.0f, 0.5f);
    board.text(board.tr("fig.centre"), {centre.x, top - 12.0f}, kRed, 11.0f, 0.5f);

    // The arrow between them.
    const float mid = area.w * 0.5f;
    board.line({mid - 22.0f, top + frame * 0.5f}, {mid + 22.0f, top + frame * 0.5f}, kMuted, 2.5f);
    board.line({mid + 10.0f, top + frame * 0.5f - 8.0f}, {mid + 22.0f, top + frame * 0.5f}, kMuted, 2.5f);
    board.line({mid + 10.0f, top + frame * 0.5f + 8.0f}, {mid + 22.0f, top + frame * 0.5f}, kMuted, 2.5f);
}

}

void draw_figure(FigureId figure, IDrawList& list, const Rect& area, const Tr& tr, const LineParams& line) {
    Board board(list, area, tr);
    board.box({0.0f, 0.0f, area.w, area.h}, kPaper, kGrid);
    switch (figure) {
    case FigureId::None: break;
    case FigureId::Line: line_figure(board, line); break;
    case FigureId::Pixels: pixels_figure(board); break;
    case FigureId::Scores: scores_figure(board); break;
    case FigureId::Softmax: softmax_figure(board); break;
    case FigureId::Loss: loss_figure(board); break;
    case FigureId::Descent: descent_figure(board); break;
    case FigureId::Network: network_figure(board); break;
    case FigureId::Relu: relu_figure(board); break;
    case FigureId::Overfit: overfit_figure(board); break;
    case FigureId::Augment: augment_figure(board); break;
    case FigureId::Normalize: normalize_figure(board); break;
    }
}

}
