#pragma once

#include "model/dream.h"

#include <atomic>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <vector>

namespace game {

// Works out the dream pictures of the neurons of one layer on background threads. The pictures come back one by one as
// they are finished, and the one the player is looking at can be moved to the front of the line.
class NeuronDreamer {
public:
    struct Picture {
        std::size_t neuron = 0;
        Pixels pixels{};
    };

    NeuronDreamer() = default;
    ~NeuronDreamer() { stop(); }

    NeuronDreamer(const NeuronDreamer&) = delete;
    NeuronDreamer& operator=(const NeuronDreamer&) = delete;

    // Drops whatever is running and starts on `neurons` of `layer`, in that order.
    void start(std::shared_ptr<const Mlp> model, std::size_t layer, std::vector<std::size_t> neurons);
    // Returns quickly: the tasks check for it between steps.
    void stop();
    // Makes `neuron` the next one worked on. Nothing happens when it is done or was not in the list.
    void prioritize(std::size_t neuron);
    // The pictures finished since the last call.
    [[nodiscard]] std::vector<Picture> collect();

private:
    static constexpr std::size_t kNone = std::numeric_limits<std::size_t>::max();

    struct Job {
        std::shared_ptr<const Mlp> model;
        std::size_t layer = 0;
        std::vector<std::size_t> neurons;
        // Indexed by neuron: set by the task that takes it, so the urgent one is not worked on twice.
        std::unique_ptr<std::atomic<bool>[]> claimed;
        std::atomic<std::size_t> cursor{0};
        std::atomic<std::size_t> urgent{kNone};
        std::atomic<bool> cancel{false};
        std::mutex mutex;
        std::vector<Picture> done;
    };

    static void work(Job& job);

    std::shared_ptr<Job> job_;
    std::vector<std::future<void>> workers_;
};

}
