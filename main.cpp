#include "audio/firFilter/AudioFile.h"
#include "audio/firFilter/runtime/LibTorchRuntime.h"
#include "audio/firFilter/runtime/NNRuntime.h"
#include "audio/firFilter/training/FIRFilterModel.h"

#include <algorithm>
#include <cmath>
#include <iostream>

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

    std::cout << "LibTorch runtime output samples: " << output.size()
              << std::endl;
    std::cout << "nn-runtime output samples: " << nnOutput.size() << std::endl;

    float maxDifference = 0.0F;
    for (size_t index = 0; index < std::min(output.size(), nnOutput.size());
         ++index) {
        maxDifference = std::max(maxDifference,
                                 std::abs(output[index] - nnOutput[index]));
    }
    std::cout << "runtime max absolute difference: " << maxDifference
              << std::endl;

    audio::firFilter::MonoAudio outputAudio;
    outputAudio.sampleRate = runtimeInput.sampleRate;
    outputAudio.samples = output;

    audio::firFilter::writeMonoWav("output.wav", outputAudio);

    audio::firFilter::MonoAudio nnOutputAudio;
    nnOutputAudio.sampleRate = runtimeInput.sampleRate;
    nnOutputAudio.samples = nnOutput;

    audio::firFilter::writeMonoWav("nnOutput.wav", nnOutputAudio);

    audio::firFilter::writeModelJson("firFilterModel.json",
                                     result.weights,
                                     result.fixture.sampleRate);

    return 0;
}
