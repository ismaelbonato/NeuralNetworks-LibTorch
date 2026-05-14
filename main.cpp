#include "audio/firFilter/AudioFile.h"
#include "audio/firFilter/runtime/LibTorchRuntime.h"
#include "audio/firFilter/runtime/NNRuntime.h"
#include "audio/firFilter/serialization/FIRFilterModelJson.h"
#include "audio/firFilter/training/FIRFilterModel.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>

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

std::string outputPath(const audio::firFilter::FIRFilterOutputConfig &config,
                       const std::string &fileName)
{
    return (std::filesystem::path(config.outputDirectory) / fileName).string();
}

} // namespace

int main()
{
    const audio::firFilter::FIRFilterConfig config{};
    const audio::firFilter::FIRFilterOutputConfig outputConfig{};

    std::filesystem::create_directories(outputConfig.outputDirectory);

    std::cout << "Training FIR model" << std::endl;
    auto result = audio::firFilter::trainFIRFilter(config);

    std::cout << "Loading runtime input: " << config.runtimeInputPath
              << std::endl;
    const auto runtimeInput = audio::firFilter::readMonoWav(
        config.runtimeInputPath);

    std::cout << "Running LibTorch runtime" << std::endl;
    audio::firFilter::LibTorchRuntime runtime(result.network);
    const auto output = runtime.inferMeasured(runtimeInput.samples);

    std::cout << "Running in-memory nn-runtime" << std::endl;
    audio::firFilter::NNRuntime nnRuntime(result.weights);
    const auto nnOutput = nnRuntime.inferMeasured(runtimeInput.samples);

    const auto modelJsonPath = outputPath(outputConfig, outputConfig.modelJson);
    std::cout << "Exporting FIR model JSON: " << modelJsonPath << std::endl;
    audio::firFilter::writeModelJson(modelJsonPath,
                                     result.weights,
                                     result.fixture.sampleRate);

    std::cout << "Loading exported FIR model JSON" << std::endl;
    const auto loadedModel = audio::firFilter::loadFIRFilterModelJsonFile(
        modelJsonPath);

    std::cout << "Running JSON-loaded nn-runtime" << std::endl;
    audio::firFilter::NNRuntime jsonRuntime(loadedModel.weights);
    const auto jsonOutput = jsonRuntime.inferMeasured(runtimeInput.samples);

    std::cout << "Verification summary" << std::endl;
    std::cout << "LibTorch runtime output samples: " << output.samples.size()
              << std::endl;
    std::cout << "nn-runtime output samples: " << nnOutput.samples.size()
              << std::endl;
    std::cout << "JSON nn-runtime output samples: "
              << jsonOutput.samples.size() << std::endl;
    std::cout << "runtime max absolute difference: "
              << maxAbsoluteDifference(output.samples, nnOutput.samples)
              << std::endl;
    std::cout << "JSON runtime max absolute difference: "
              << maxAbsoluteDifference(output.samples, jsonOutput.samples)
              << std::endl;
    std::cout << "LibTorch runtime inference time: "
              << output.elapsedMilliseconds << " ms" << std::endl;
    std::cout << "nn-runtime inference time: " << nnOutput.elapsedMilliseconds
              << " ms" << std::endl;
    std::cout << "JSON nn-runtime inference time: "
              << jsonOutput.elapsedMilliseconds << " ms" << std::endl;

    audio::firFilter::MonoAudio outputAudio;
    outputAudio.sampleRate = runtimeInput.sampleRate;
    outputAudio.samples = output.samples;

    const auto libTorchOutputPath = outputPath(outputConfig,
                                               outputConfig.libTorchOutputWav);
    std::cout << "Writing LibTorch output WAV: " << libTorchOutputPath
              << std::endl;
    audio::firFilter::writeMonoWav(libTorchOutputPath, outputAudio);

    audio::firFilter::MonoAudio nnOutputAudio;
    nnOutputAudio.sampleRate = runtimeInput.sampleRate;
    nnOutputAudio.samples = nnOutput.samples;

    const auto nnRuntimeOutputPath = outputPath(
        outputConfig,
        outputConfig.nnRuntimeOutputWav);
    std::cout << "Writing nn-runtime output WAV: " << nnRuntimeOutputPath
              << std::endl;
    audio::firFilter::writeMonoWav(nnRuntimeOutputPath, nnOutputAudio);

    return 0;
}
