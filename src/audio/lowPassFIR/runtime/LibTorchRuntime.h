#pragma once

#include "audio/lowPassFIR/training/LowPassFIRModel.h"

#include <memory>
#include <ostream>
#include <vector>

namespace audio::lowPassFIR {

class LibTorchRuntime
{
public:
    explicit LibTorchRuntime(std::shared_ptr<LowPassFIRNetwork> trainedNetwork);

    std::vector<float> infer(const std::vector<float> &input);
    std::vector<float> printInferenceVectors(const std::vector<float> &input,
                                             std::ostream &output);

private:
    std::shared_ptr<LowPassFIRNetwork> network;
};

} // namespace audio::lowPassFIR
