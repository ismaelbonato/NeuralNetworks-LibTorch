#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <torch/torch.h>
#include <vector>

namespace audio::lowPassFIR {

const std::string audioSample
    = "../nn-training-assets/audio/lowPassFIR/monoWhiteNoise.wav";
const std::string audioTarget
    = "../nn-training-assets/audio/lowPassFIR/monoWhiteNoiseTarget.wav";

constexpr size_t inputChannels = 1;
constexpr size_t outputChannels = 1;
constexpr size_t sampleCount = 256;
constexpr size_t kernelSize = 9;
constexpr size_t trainingEpochs = 600;

struct TrainingFixture
{
    // One synthetic mono signal and the filtered signal it should learn.
    std::vector<float> input;
    std::vector<float> target;
    std::vector<float> targetKernel;
};

struct ConvolutionWeights
{
    // Plain exported weights for building the same FIR layer in another runtime.
    size_t inputChannelCount = inputChannels;
    size_t outputChannelCount = outputChannels;
    size_t inputLength = sampleCount;
    size_t kernelLength = kernelSize;
    size_t stride = 1;
    size_t padding = 0;
    std::vector<float> weights;
    std::vector<float> biases;
};

struct TrainingResult
{
    // Keep the trained model and plain weights so training and runtime can split.
    std::shared_ptr<struct LowPassFIRNetwork> network;
    TrainingFixture fixture;
    ConvolutionWeights weights;
    float initialLoss = 0.0F;
    float finalLoss = 0.0F;
};

struct LowPassFIRNetwork : torch::nn::Module
{
    LowPassFIRNetwork();

    torch::Tensor forward(const torch::Tensor &input);

    // Conv1d is a learnable FIR kernel when channel counts are 1 and bias is off.
    torch::nn::Conv1d convolution{nullptr};
};

TrainingFixture makeTrainingFixture();

ConvolutionWeights exportWeights(const LowPassFIRNetwork &network);

TrainingResult trainLowPassFIR();

} // namespace audio::lowPassFIR
