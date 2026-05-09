#pragma once

#include "audio/lowPassFIR/training/LowPassFIRModel.h"

#include <vector>

namespace audio::lowPassFIR {

class NNRuntime
{
public:
    explicit NNRuntime(ConvolutionWeights weights);

    std::vector<float> infer(const std::vector<float> &input) const;

private:
    ConvolutionWeights weights;
};

} // namespace audio::lowPassFIR
