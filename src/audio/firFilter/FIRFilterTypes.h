#pragma once

#include <cstddef>
#include <string>
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

struct FIRFilterOutputConfig
{
    std::string outputDirectory = "output/fir-filter";
    std::string libTorchOutputWav = "libtorch-output.wav";
    std::string nnRuntimeOutputWav = "nn-runtime-output.wav";
    std::string modelJson = "fir-filter-model.json";
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

} // namespace audio::firFilter
