#include "recognizer_texts.h"

#include <algorithm>
#include <format>

namespace game {

namespace {

std::string percent(float fraction) {
    return std::format("{:.2f}%", fraction * 100.0f);
}

}

void RecognizerTexts::set_prediction(std::optional<std::size_t> digit) {
    prediction_ = digit;
    view_.predictionText = digit ? tr_("text.prediction", {{"digit", std::to_string(*digit)}})
                                 : tr_("text.prediction_none");
}

void RecognizerTexts::set_training(bool training) {
    training_ = training;
    view_.trainButtonText = tr_(training ? "btn.training" : "btn.train");
}

void RecognizerTexts::set_model(ModelInfo model) {
    model_ = std::move(model);
    render_model();
}

void RecognizerTexts::waiting_for_scores(bool possible) {
    score_state_ = possible ? ScoreState::Waiting : ScoreState::Unavailable;
    render_scores();
}

void RecognizerTexts::scoring(float progress) {
    score_state_ = ScoreState::Scoring;
    progress_ = progress;
    view_.testText = tr_("test.line", {{"value", tr_("test.scoring", {{"percent", std::format("{:.0f}", progress * 100.0f)}})}});
}

void RecognizerTexts::set_scores(const Scores& scores) {
    scores_ = scores;
    score_state_ = ScoreState::Done;
    render_scores();
}

void RecognizerTexts::set_last_training(LastTraining last) {
    last_ = std::move(last);
    render_last_training();
}

void RecognizerTexts::render() {
    set_prediction(prediction_);
    set_training(training_);
    render_model();
    render_scores();
    render_last_training();
}

void RecognizerTexts::render_model() {
    if (!model_) {
        view_.modelNameText = tr_("model.none");
        view_.modelArchText = tr_("model.arch", {{"value", "-"}});
        view_.modelLayersText = tr_("model.layers", {{"value", "-"}});
        view_.modelEpochsText = tr_("model.epochs", {{"value", "-"}});
        return;
    }
    const std::vector<std::size_t>& hidden = model_->hidden;
    view_.modelNameText = tr_("model.name", {{"name", model_->name}});
    view_.modelArchText = tr_("model.arch", {{"value", model_->architecture}});
    if (hidden.empty()) {
        view_.modelLayersText = tr_("model.layers_linear");
    } else if (std::ranges::all_of(hidden, [&](std::size_t size) { return size == hidden.front(); })) {
        view_.modelLayersText = tr_("model.layers_equal",
                {{"count", std::to_string(hidden.size())}, {"neurons", std::to_string(hidden.front())}});
    } else {
        std::string sizes;
        for (const std::size_t size : hidden) {
            sizes += (sizes.empty() ? "" : ", ") + std::to_string(size);
        }
        view_.modelLayersText = tr_("model.layers_mixed", {{"sizes", sizes}});
    }
    view_.modelEpochsText = tr_("model.epochs",
            {{"value", model_->epochs > 0 ? std::to_string(model_->epochs) : tr_("model.epochs_unknown")}});
}

void RecognizerTexts::render_scores() {
    const auto line = [&](const char* key, const std::string& value) { return tr_(key, {{"value", value}}); };
    switch (score_state_) {
    case ScoreState::Unavailable:
    case ScoreState::Waiting: {
        const std::string mark = score_state_ == ScoreState::Waiting ? "..." : "-";
        view_.testText = line("test.line", mark);
        view_.testMutatedText = line("test.mutated", mark);
        view_.hardestDigitText = line("digits.hardest", mark);
        view_.easiestDigitText = line("digits.easiest", mark);
        view_.perDigitText = line("digits.per_digit", mark);
        return;
    }
    case ScoreState::Scoring:
        scoring(progress_);
        view_.testMutatedText = line("test.mutated", "...");
        view_.hardestDigitText = line("digits.hardest", "...");
        view_.easiestDigitText = line("digits.easiest", "...");
        view_.perDigitText = line("digits.per_digit", "...");
        return;
    case ScoreState::Done:
        break;
    }

    view_.testText = line("test.line", percent(scores_.test));
    view_.testMutatedText = line("test.mutated", percent(scores_.mutated));

    // The digit MNIST test trips the model on most and least, by the share of that digit it gets wrong.
    std::array<float, kNumClasses> error_rate{};
    std::string per_digit;
    for (std::size_t digit = 0; digit < kNumClasses; ++digit) {
        const std::size_t total = scores_.total[digit];
        const std::size_t wrong = total - scores_.correct[digit];
        error_rate[digit] = total == 0 ? 0.0f : static_cast<float>(wrong) / static_cast<float>(total);
        per_digit += std::format("{}: {:.1f}%{}", digit, (1.0f - error_rate[digit]) * 100.0f,
                digit + 1 < kNumClasses ? ",  " : "");
    }
    const auto describe = [&](std::size_t digit) {
        return tr_("digits.detail",
                {{"digit", std::to_string(digit)},
                        {"rate", std::format("{:.1f}", error_rate[digit] * 100.0f)},
                        {"wrong", std::to_string(scores_.total[digit] - scores_.correct[digit])},
                        {"total", std::to_string(scores_.total[digit])}});
    };
    const auto [easiest, hardest] = std::ranges::minmax_element(error_rate);
    view_.hardestDigitText = line("digits.hardest", describe(static_cast<std::size_t>(hardest - error_rate.begin())));
    view_.easiestDigitText = line("digits.easiest", describe(static_cast<std::size_t>(easiest - error_rate.begin())));
    view_.perDigitText = line("digits.per_digit", per_digit);
}

void RecognizerTexts::render_last_training() {
    if (!last_) {
        view_.lastDatasetText = tr_("last.none");
        view_.lastTrainText.set({});
        view_.lastLossText.set({});
        view_.lastSavedText.set({});
        return;
    }
    view_.lastDatasetText = tr_("last.dataset",
            {{"name", last_->dataset}, {"samples", std::to_string(last_->samples)}});
    view_.lastTrainText = last_->scored
            ? tr_("last.train", {{"acc", percent(last_->train)}, {"mutated", percent(last_->train_mutated)}})
            : tr_("last.train", {{"acc", "..."}, {"mutated", "..."}});
    view_.lastLossText = tr_("last.loss", {{"loss", std::format("{:.4f}", last_->loss)}});
    view_.lastSavedText = last_->save_error.empty() ? tr_("last.saved", {{"file", last_->file}})
                                                    : tr_("last.not_saved", {{"detail", last_->save_error}});
}

}
