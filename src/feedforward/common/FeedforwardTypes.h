#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace feedforward {

constexpr float outputTolerance = 0.0001F;
constexpr size_t benchmarkIterations = 100000;

struct DenseLayerWeights
{
    std::string name;
    size_t inputSize = 0;
    size_t outputSize = 0;
    std::vector<float> weights;
    std::vector<float> biases;
};

struct FeedforwardWeights
{
    std::vector<DenseLayerWeights> layers;
};

struct InferenceSample
{
    std::vector<float> input;
};

struct OutputComparison
{
    std::vector<float> input;
    float expected = 0.0F;
    float actual = 0.0F;
    float difference = 0.0F;
    bool withinTolerance = false;
};

std::vector<InferenceSample> makeXorSamples();

} // namespace feedforward
