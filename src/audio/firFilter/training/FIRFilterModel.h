#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <torch/torch.h>
#include <vector>

namespace audio::firFilter {

const std::string defaultTrainingInputPath
    = "../nn-training-assets/audio/lowPassFIR/monoWhiteNoise.wav";
const std::string defaultTrainingTargetPath
    = "../nn-training-assets/audio/lowPassFIR/monoWhiteNoiseLowPass2k.wav";
const std::string defaultRuntimeInputPath
    = "../nn-training-assets/audio/lowPassFIR/song.wav";

constexpr size_t inputChannels = 1;
constexpr size_t outputChannels = 1;

struct FIRFilterConfig
{
    std::string trainingInputPath = defaultTrainingInputPath;
    std::string trainingTargetPath = defaultTrainingTargetPath;
    std::string runtimeInputPath = defaultRuntimeInputPath;
    size_t kernelSize = 101;
    size_t trainingEpochs = 600;
    size_t targetAlignmentOffset = 50;
};

struct TrainingFixture
{
    // One source signal and the filtered signal it should learn.
    std::vector<float> input;
    std::vector<float> target;
    uint32_t sampleRate = 44100;
};

struct ConvolutionWeights
{
    // Plain exported weights for building the same FIR layer in another runtime.
    size_t inputChannelCount = inputChannels;
    size_t outputChannelCount = outputChannels;
    size_t inputLength = 0;
    size_t kernelLength = 0;
    size_t stride = 1;
    size_t padding = 0;
    std::vector<float> weights;
    std::vector<float> biases;
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
