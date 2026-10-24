#pragma once

#include "model/dream.h"
#include "model/mlp.h"
#include "view/prototype_paint.h"

#include <array>
#include <atomic>
#include <future>
#include <memory>
#include <optional>

namespace game {

// Works out, in the background, what a model takes for the ideal picture of each digit and hands every picture to the
// paint as soon as it is ready: one task per digit, all reading the same model.
class DigitDreamer {
public:
    explicit DigitDreamer(PrototypePaint& paint)
        : paint_(paint) {}

    ~DigitDreamer() { stop(); }

    DigitDreamer(const DigitDreamer&) = delete;
    DigitDreamer& operator=(const DigitDreamer&) = delete;

    // Drops the pictures of the previous model and starts on `model`.
    void start(std::shared_ptr<const Mlp> model);
    // Cancels the tasks still running and waits for them.
    void stop();
    // Once per frame on the main thread.
    void collect();

private:
    PrototypePaint& paint_;
    std::shared_ptr<std::atomic<bool>> cancel_;
    std::array<std::future<std::optional<Pixels>>, kNumClasses> dreams_;
};

}
