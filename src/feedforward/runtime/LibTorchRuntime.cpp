#include "LibTorchRuntime.h"

#include <stdexcept>
#include <utility>

namespace feedforward {

LibTorchRuntime::LibTorchRuntime(std::shared_ptr<FeedforwardNetwork> newNetwork)
    : network(std::move(newNetwork))
{
    if (!network) {
        throw std::invalid_argument("LibTorchRuntime requires a network");
    }
    network->eval();
}

const char *LibTorchRuntime::name() const
{
    return "LibTorch";
}

LibTorchRuntime::InputBatch LibTorchRuntime::prepareInputs(
    const std::vector<InferenceSample> &samples) const
{
    InputBatch inputs;
    inputs.reserve(samples.size());
    for (const auto &sample : samples) {
        inputs.emplace_back(torch::tensor(sample.input));
    }
    return inputs;
}

float LibTorchRuntime::infer(const torch::Tensor &input)
{
    torch::NoGradGuard noGrad;
    return network->forward(input).item<float>();
}

float LibTorchRuntime::runBatch(const InputBatch &inputs)
{
    torch::NoGradGuard noGrad;
    float total = 0.0F;
    for (const auto &input : inputs) {
        total += network->forward(input).item<float>();
    }
    return total;
}

} // namespace feedforward
