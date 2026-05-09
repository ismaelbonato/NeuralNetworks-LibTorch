#include "audio/lowPassFIR/AudioFile.h"
#include "audio/lowPassFIR/runtime/LibTorchRuntime.h"
#include "audio/lowPassFIR/runtime/NNRuntime.h"
#include "audio/lowPassFIR/training/LowPassFIRModel.h"

#include <algorithm>
#include <cmath>
#include <iostream>

int main()
{
    auto result = audio::lowPassFIR::trainLowPassFIR();

    result.fixture.input
        = audio::lowPassFIR::readMonoWav(audio::lowPassFIR::audioTest).samples;

    // libtorch runtime
    audio::lowPassFIR::LibTorchRuntime runtime(result.network);
    const auto output = runtime.printInferenceVectors(result.fixture.input,
                                                      std::cout);

    // nn-runtime runtime
    audio::lowPassFIR::NNRuntime nnRuntime(result.weights);
    const auto nnOutput = nnRuntime.infer(result.fixture.input);

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

    audio::lowPassFIR::MonoAudio outputAudio;
    outputAudio.sampleRate = 44100;
    outputAudio.samples = output;

    audio::lowPassFIR::writeMonoWav("output.wav", outputAudio);

    audio::lowPassFIR::MonoAudio nnOutputAudio;
    nnOutputAudio.sampleRate = 44100;
    nnOutputAudio.samples = nnOutput;

    audio::lowPassFIR::writeMonoWav("nnOutput.wav", nnOutputAudio);

    return 0;
}
