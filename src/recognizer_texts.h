#pragma once

#include "loc/tr.h"
#include "model/digits.h"
#include "view/recognizer_view_model.h"

#include <array>
#include <optional>
#include <string>
#include <vector>

namespace game {

// Writes the texts of the Recognize and Train tabs that show the state of the program: the prediction, the loaded
// model, its scores, the last training. The state is kept here as numbers and names, never as finished sentences, so
// `render` can write every text again in another language.
class RecognizerTexts {
public:
    struct ModelInfo {
        std::string name;
        std::string architecture;  // "784-64-10"
        std::vector<std::size_t> hidden;
        unsigned epochs = 0;  // 0: not known
    };

    struct Scores {
        float test = 0.0f;
        float mutated = 0.0f;
        std::array<std::size_t, kNumClasses> total{};
        std::array<std::size_t, kNumClasses> correct{};
    };

    struct LastTraining {
        std::string dataset;
        std::size_t samples = 0;
        float loss = 0.0f;
        std::string file;
        std::string save_error;  // empty: saved
        bool scored = false;
        float train = 0.0f;
        float train_mutated = 0.0f;
    };

    RecognizerTexts(RecognizerViewModel& view, const Tr& tr)
        : view_(view)
        , tr_(tr) {}

    void set_prediction(std::optional<std::size_t> digit);
    void set_training(bool training);
    void set_model(ModelInfo model);
    // The model's scores are not known yet: they are being worked out (`progress` 0..1) or cannot be (no dataset).
    void waiting_for_scores(bool possible);
    void scoring(float progress);
    void set_scores(const Scores& scores);
    void set_last_training(LastTraining last);
    [[nodiscard]] LastTraining* last_training() { return last_ ? &*last_ : nullptr; }

    void render();

private:
    enum class ScoreState { Unavailable, Waiting, Scoring, Done };

    void render_model();
    void render_scores();
    void render_last_training();

    RecognizerViewModel& view_;
    const Tr& tr_;

    std::optional<std::size_t> prediction_;
    bool training_ = false;
    std::optional<ModelInfo> model_;
    ScoreState score_state_ = ScoreState::Unavailable;
    float progress_ = 0.0f;
    Scores scores_;
    std::optional<LastTraining> last_;
};

}
