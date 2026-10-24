#include "model/neuron_dreamer.h"

#include <algorithm>
#include <thread>

namespace game {

void NeuronDreamer::start(std::shared_ptr<const Mlp> model, std::size_t layer, std::vector<std::size_t> neurons) {
    stop();
    if (neurons.empty()) {
        return;
    }
    job_ = std::make_shared<Job>();
    job_->layer = layer;
    job_->claimed = std::make_unique<std::atomic<bool>[]>(model->layer_size(layer));
    job_->model = std::move(model);
    job_->neurons = std::move(neurons);

    // Leave a core for the window and for training, which runs on the job system at the same time.
    const unsigned cores = std::thread::hardware_concurrency();
    const unsigned count = cores > 2 ? std::min(cores - 1, 8u) : 1u;
    for (unsigned i = 0; i < count; ++i) {
        workers_.push_back(std::async(std::launch::async, [job = job_] { work(*job); }));
    }
}

void NeuronDreamer::stop() {
    if (job_) {
        job_->cancel.store(true);
    }
    for (auto& worker : workers_) {
        worker.wait();
    }
    workers_.clear();
    job_.reset();
}

void NeuronDreamer::prioritize(std::size_t neuron) {
    if (job_) {
        job_->urgent.store(neuron);
    }
}

std::vector<NeuronDreamer::Picture> NeuronDreamer::collect() {
    std::vector<Picture> pictures;
    if (job_) {
        const std::lock_guard lock(job_->mutex);
        pictures.swap(job_->done);
    }
    return pictures;
}

void NeuronDreamer::work(Job& job) {
    while (!job.cancel.load(std::memory_order_relaxed)) {
        std::size_t neuron = job.urgent.exchange(kNone);
        if (neuron == kNone || neuron >= job.model->layer_size(job.layer) || job.claimed[neuron].exchange(true)) {
            const std::size_t next = job.cursor.fetch_add(1);
            if (next >= job.neurons.size()) {
                return;
            }
            neuron = job.neurons[next];
            if (job.claimed[neuron].exchange(true)) {
                continue;
            }
        }
        auto pixels = dream_neuron(*job.model, job.layer, neuron, job.cancel);
        if (!pixels) {
            return;
        }
        const std::lock_guard lock(job.mutex);
        job.done.push_back({neuron, *pixels});
    }
}

}
