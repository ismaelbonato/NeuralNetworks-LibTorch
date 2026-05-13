#pragma once

#include "audio/firFilter/FIRFilterTypes.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <torch/torch.h>
#include <vector>

namespace audio::firFilter {

struct TrainingFixture
{
    // One source signal and the filtered signal it should learn.
    std::vector<float> input;
    std::vector<float> target;
    uint32_t sampleRate = 44100;
};

struct TrainingResult
{
    // Keep the trained model and plain weights so training and runtime can split.
    std::shared_ptr<struct FIRFilterNetwork> network;
    TrainingFixture fixture;
    ConvolutionWeights weights;
    float initialLoss = 0.0F;
    float finalLoss = 0.0F;
};

struct FIRFilterNetwork : torch::nn::Module
{
    explicit FIRFilterNetwork(const FIRFilterConfig &config);

    torch::Tensor forward(const torch::Tensor &input);

    // Conv1d is a learnable FIR kernel when channel counts are 1 and bias is off.
    torch::nn::Conv1d convolution{nullptr};
};

TrainingFixture loadTrainingFixture(const FIRFilterConfig &config);

ConvolutionWeights exportWeights(const FIRFilterNetwork &network,
                                 const FIRFilterConfig &config,
                                 size_t inputLength);

TrainingResult trainFIRFilter(const FIRFilterConfig &config = {});

void writeModelJson(const std::string &path,
                    const ConvolutionWeights &weights,
                    uint32_t sampleRate);

} // namespace audio::firFilter
