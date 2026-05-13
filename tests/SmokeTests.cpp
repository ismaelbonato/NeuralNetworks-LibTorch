#include "audio/firFilter/runtime/NNRuntime.h"
#include "audio/firFilter/serialization/FIRFilterModelJson.h"

#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr const char *validModelJson = R"json(
{
  "format": "nn-training.fir-filter.v1",
  "sampleRate": 44100,
  "architecture": {
    "type": "Conv1D",
    "inputChannels": 1,
    "outputChannels": 1,
    "kernelSize": 3,
    "stride": 1,
    "padding": 0,
    "activation": "identity"
  },
  "parameters": {
    "weights": [0.25, 0.5, 0.25],
    "biases": [0.0]
  }
}
)json";

bool almostEqual(const float lhs, const float rhs)
{
    return std::abs(lhs - rhs) <= 0.000001F;
}

void require(const bool condition, const std::string &message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void requireThrows(const std::function<void()> &action,
                   const std::string &message)
{
    try {
        action();
    } catch (const std::runtime_error &) {
        return;
    }

    throw std::runtime_error(message);
}

std::string replaceFirst(std::string value,
                         const std::string &from,
                         const std::string &to)
{
    const auto position = value.find(from);
    if (position == std::string::npos) {
        throw std::runtime_error("test fixture replacement failed");
    }
    value.replace(position, from.size(), to);
    return value;
}

void testLoadsValidFIRFilterJson()
{
    const auto model = audio::firFilter::loadFIRFilterModelJson(validModelJson);

    require(model.sampleRate == 44100, "sample rate was not loaded");
    require(model.weights.inputChannelCount == 1, "input channel count mismatch");
    require(model.weights.outputChannelCount == 1,
            "output channel count mismatch");
    require(model.weights.kernelLength == 3, "kernel size mismatch");
    require(model.weights.stride == 1, "stride mismatch");
    require(model.weights.padding == 0, "padding mismatch");
    require(model.weights.weights.size() == 3, "weight count mismatch");
    require(model.weights.biases.size() == 1, "bias count mismatch");
    require(almostEqual(model.weights.weights.at(0), 0.25F),
            "first weight mismatch");
}

void testRejectsInvalidFIRFilterJson()
{
    requireThrows([] {
        audio::firFilter::loadFIRFilterModelJson("{");
    }, "malformed JSON should throw");

    requireThrows([] {
        audio::firFilter::loadFIRFilterModelJson(
            replaceFirst(validModelJson,
                         "nn-training.fir-filter.v1",
                         "wrong-format"));
    }, "wrong format should throw");

    requireThrows([] {
        audio::firFilter::loadFIRFilterModelJson(
            replaceFirst(validModelJson, "identity", "relu"));
    }, "unsupported activation should throw");

    requireThrows([] {
        audio::firFilter::loadFIRFilterModelJson(
            replaceFirst(validModelJson,
                         "\"weights\": [0.25, 0.5, 0.25]",
                         "\"weights\": [0.25, 0.5]"));
    }, "weight count mismatch should throw");
}

void testJsonLoadedRuntimeMatchesInMemoryRuntime()
{
    const auto model = audio::firFilter::loadFIRFilterModelJson(validModelJson);

    audio::firFilter::ConvolutionWeights inMemoryWeights;
    inMemoryWeights.inputChannelCount = 1;
    inMemoryWeights.outputChannelCount = 1;
    inMemoryWeights.kernelLength = 3;
    inMemoryWeights.stride = 1;
    inMemoryWeights.padding = 0;
    inMemoryWeights.weights = {0.25F, 0.5F, 0.25F};
    inMemoryWeights.biases = {0.0F};

    audio::firFilter::NNRuntime inMemoryRuntime(inMemoryWeights);
    audio::firFilter::NNRuntime jsonRuntime(model.weights);

    const std::vector<float> input = {0.0F, 1.0F, 2.0F, 3.0F, 4.0F};
    const auto inMemoryOutput = inMemoryRuntime.infer(input);
    const auto jsonOutput = jsonRuntime.infer(input);

    require(inMemoryOutput.size() == jsonOutput.size(),
            "runtime output size mismatch");
    for (size_t index = 0; index < inMemoryOutput.size(); ++index) {
        require(almostEqual(inMemoryOutput.at(index), jsonOutput.at(index)),
                "runtime output value mismatch");
    }
}

} // namespace

int main()
{
    try {
        testLoadsValidFIRFilterJson();
        testRejectsInvalidFIRFilterJson();
        testJsonLoadedRuntimeMatchesInMemoryRuntime();
    } catch (const std::exception &error) {
        std::cerr << "Smoke test failed: " << error.what() << '\n';
        return 1;
    }

    std::cout << "Smoke test passed" << '\n';
    return 0;
}
