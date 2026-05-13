#include "audio/firFilter/AudioFile.h"
#include "audio/firFilter/runtime/LibTorchRuntime.h"
#include "audio/firFilter/runtime/NNRuntime.h"
#include "audio/firFilter/serialization/FIRFilterModelJson.h"
#include "audio/firFilter/training/FIRFilterModel.h"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace {

float maxAbsoluteDifference(const std::vector<float> &expected,
                            const std::vector<float> &actual)
{
    float maxDifference = 0.0F;
    for (size_t index = 0; index < std::min(expected.size(), actual.size());
         ++index) {
        maxDifference = std::max(maxDifference,
                                 std::abs(expected[index] - actual[index]));
    }
    return maxDifference;
}

} // namespace

int main()
{
    const audio::firFilter::FIRFilterConfig config{};
    auto result = audio::firFilter::trainFIRFilter(config);

    const auto runtimeInput = audio::firFilter::readMonoWav(
        config.runtimeInputPath);

    // libtorch runtime
    audio::firFilter::LibTorchRuntime runtime(result.network);
    const auto output = runtime.infer(runtimeInput.samples);

    // nn-runtime runtime
    audio::firFilter::NNRuntime nnRuntime(result.weights);
    const auto nnOutput = nnRuntime.infer(runtimeInput.samples);

    audio::firFilter::writeModelJson("firFilterModel.json",
                                     result.weights,
                                     result.fixture.sampleRate);

    const auto loadedModel = audio::firFilter::loadFIRFilterModelJsonFile(
        "firFilterModel.json");
    audio::firFilter::NNRuntime jsonRuntime(loadedModel.weights);
    const auto jsonOutput = jsonRuntime.infer(runtimeInput.samples);

    std::cout << "LibTorch runtime output samples: " << output.size()
              << std::endl;
    std::cout << "nn-runtime output samples: " << nnOutput.size() << std::endl;
    std::cout << "JSON nn-runtime output samples: " << jsonOutput.size()
              << std::endl;
    std::cout << "runtime max absolute difference: "
              << maxAbsoluteDifference(output, nnOutput) << std::endl;
    std::cout << "JSON runtime max absolute difference: "
              << maxAbsoluteDifference(output, jsonOutput)
              << std::endl;

    audio::firFilter::MonoAudio outputAudio;
    outputAudio.sampleRate = runtimeInput.sampleRate;
    outputAudio.samples = output;

    audio::firFilter::writeMonoWav("output.wav", outputAudio);

    audio::firFilter::MonoAudio nnOutputAudio;
    nnOutputAudio.sampleRate = runtimeInput.sampleRate;
    nnOutputAudio.samples = nnOutput;

    audio::firFilter::writeMonoWav("nnOutput.wav", nnOutputAudio);

    return 0;
}
