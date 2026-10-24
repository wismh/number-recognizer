#include "digit_dreamer.h"

#include <chrono>

namespace game {

void DigitDreamer::start(std::shared_ptr<const Mlp> model) {
    stop();
    paint_.reset();
    cancel_ = std::make_shared<std::atomic<bool>>(false);
    for (std::size_t digit = 0; digit < kNumClasses; ++digit) {
        dreams_[digit] = std::async(std::launch::async,
                [model, cancel = cancel_, digit] { return dream_digit(*model, digit, *cancel); });
    }
}

void DigitDreamer::stop() {
    if (cancel_) {
        cancel_->store(true);
    }
    for (auto& dream : dreams_) {
        if (dream.valid()) {
            dream.wait();  // returns quickly: the task checks the flag every step
            dream = {};
        }
    }
}

void DigitDreamer::collect() {
    for (std::size_t digit = 0; digit < kNumClasses; ++digit) {
        auto& dream = dreams_[digit];
        if (dream.valid() && dream.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
            if (const auto pixels = dream.get()) {
                paint_.set(digit, *pixels);
            }
        }
    }
}

}
