#pragma once

#include "loc/tr.h"
#include "model/augment.h"
#include "model/labeled_images.h"
#include "model/mlp.h"
#include "model/parallel.h"
#include "model/training_run.h"
#include "status_line.h"

#include <memory>
#include <optional>
#include <string>

namespace game {

// One training from start to finish as the user sees it: advances the run a slice per frame, says how far it is on the
// status line, and hands the trained model over once every epoch is done.
class TrainingSession {
public:
    struct Result {
        Mlp model;  // epochs_trained() includes this training
        std::string dataset;
        std::shared_ptr<const LabeledImages> samples;
        float loss = 0.0f;
        std::string file;  // the name to save the model under
    };

    TrainingSession(StatusLine& status, Parallel parallel)
        : status_(status)
        , parallel_(std::move(parallel)) {}

    [[nodiscard]] bool active() const { return run_.has_value(); }

    // `label` is the short name of the architecture for the file name.
    void start(Mlp model, const std::string& label, std::string dataset, std::shared_ptr<const LabeledImages> samples,
            unsigned epochs, Augmentation augmentation);
    // Once per frame on the main thread; returns the result in the frame the last epoch ends.
    [[nodiscard]] std::optional<Result> update();

private:
    void show_progress();

    StatusLine& status_;
    Parallel parallel_;
    std::optional<TrainingRun> run_;
    std::string dataset_;
    std::string file_;
    std::shared_ptr<const LabeledImages> samples_;
};

}
