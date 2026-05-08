#pragma once

#include "benchmark/feedforward/common/FeedforwardTypes.h"

#include <cstdint>
#include <torch/torch.h>

namespace feedforward {

struct FeedforwardNetwork : torch::nn::Module
{
    FeedforwardNetwork();

    torch::Tensor forward(const torch::Tensor &input);

    torch::nn::Linear inputLayer{nullptr};
    torch::nn::Linear outputLayer{nullptr};
};

FeedforwardWeights exportWeights(const FeedforwardNetwork &network);

} // namespace feedforward
