#pragma once

#include "feedforward/common/FeedforwardTypes.h"
#include "feedforward/training/FeedforwardModel.h"

#include <memory>
#include <torch/torch.h>
#include <vector>

namespace feedforward {

class LibTorchRuntime
{
public:
    using InputBatch = std::vector<torch::Tensor>;

    explicit LibTorchRuntime(std::shared_ptr<FeedforwardNetwork> newNetwork);

    const char *name() const;
    InputBatch prepareInputs(const std::vector<InferenceSample> &samples) const;
    float infer(const torch::Tensor &input);
    float runBatch(const InputBatch &inputs);

private:
    std::shared_ptr<FeedforwardNetwork> network;
};

} // namespace feedforward
