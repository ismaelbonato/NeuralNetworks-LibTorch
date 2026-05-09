#include "audio/lowPassFIR/AudioFile.h"
#include "audio/lowPassFIR/runtime/LibTorchRuntime.h"
#include "audio/lowPassFIR/training/LowPassFIRModel.h"

#include <iostream>

int main()
{
    auto result = audio::lowPassFIR::trainLowPassFIR();
    audio::lowPassFIR::LibTorchRuntime runtime(result.network);
    const auto output = runtime.infer(result.fixture.input);

    std::cout << "LibTorch runtime output samples: " << output.size()
              << std::endl;

    audio::lowPassFIR::MonoAudio outputAudio;
    outputAudio.sampleRate = 44100;
    outputAudio.samples = output;

    audio::lowPassFIR::writeMonoWav("output.wav", outputAudio);

    return 0;
}
