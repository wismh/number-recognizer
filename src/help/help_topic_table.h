#pragma once

// Generated from the help topic sources: do not edit by hand.

#include "help/help_figures.h"

#include <array>
#include <cstddef>

namespace game {

enum class HelpOpen { None, Recognize, Train, Datasets };

struct HelpTopicInfo {
    const char* section;
    const char* title;
    std::array<FigureId, kFiguresPerTopic> figures;
    std::size_t figure_count;
    HelpOpen open;
};

inline constexpr std::size_t kHelpTopicCount = 17;

inline constexpr std::array<HelpTopicInfo, kHelpTopicCount> kHelpTopics{{
    {"help.section.basics", "help.intro.title", {FigureId::None, FigureId::None}, 0, HelpOpen::Recognize},
    {"help.section.basics", "help.linear.title", {FigureId::Line, FigureId::None}, 1, HelpOpen::None},
    {"help.section.basics", "help.image.title", {FigureId::Pixels, FigureId::None}, 1, HelpOpen::Datasets},
    {"help.section.basics", "help.dot.title", {FigureId::None, FigureId::None}, 0, HelpOpen::None},
    {"help.section.classifier", "help.scores.title", {FigureId::Scores, FigureId::None}, 1, HelpOpen::None},
    {"help.section.classifier", "help.softmax.title", {FigureId::Softmax, FigureId::None}, 1, HelpOpen::None},
    {"help.section.classifier", "help.loss.title", {FigureId::Loss, FigureId::None}, 1, HelpOpen::None},
    {"help.section.training", "help.gradient.title", {FigureId::Descent, FigureId::None}, 1, HelpOpen::None},
    {"help.section.training", "help.batches.title", {FigureId::None, FigureId::None}, 0, HelpOpen::Train},
    {"help.section.network", "help.hidden.title", {FigureId::Network, FigureId::Relu}, 2, HelpOpen::Train},
    {"help.section.network", "help.backprop.title", {FigureId::None, FigureId::None}, 0, HelpOpen::None},
    {"help.section.network", "help.init.title", {FigureId::None, FigureId::None}, 0, HelpOpen::None},
    {"help.section.data", "help.mnist.title", {FigureId::Overfit, FigureId::None}, 1, HelpOpen::None},
    {"help.section.data", "help.augment.title", {FigureId::Augment, FigureId::None}, 1, HelpOpen::Train},
    {"help.section.data", "help.datasets.title", {FigureId::Normalize, FigureId::None}, 1, HelpOpen::Datasets},
    {"help.section.results", "help.metrics.title", {FigureId::None, FigureId::None}, 0, HelpOpen::None},
    {"help.section.results", "help.ideal.title", {FigureId::None, FigureId::None}, 0, HelpOpen::Recognize},
}};

}
