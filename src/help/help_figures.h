#pragma once

#include "loc/tr.h"

#include <engine/ui/draw_list.h>

#include <cstddef>

namespace game {

// The pictures of the help topics.
enum class FigureId {
    None,
    Line,
    Pixels,
    Scores,
    Softmax,
    Loss,
    Descent,
    Network,
    Relu,
    Overfit,
    Augment,
    Normalize,
};

// Most pictures one topic shows.
inline constexpr std::size_t kFiguresPerTopic = 2;

// What the reader can change in a picture: the weight and the bias of the line.
struct LineParams {
    float w = 0.5f;
    float b = 1.0f;
};

// Draws `figure` into `area` (the content box of the element it is painted on); the words in it come from `tr`.
void draw_figure(FigureId figure, engine::ui::IDrawList& list, const engine::render::Rect& area, const Tr& tr,
        const LineParams& line);

}
