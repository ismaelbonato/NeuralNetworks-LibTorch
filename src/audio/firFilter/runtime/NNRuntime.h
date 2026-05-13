#pragma once

#include "audio/firFilter/FIRFilterTypes.h"

#include <vector>

namespace audio::firFilter {

class NNRuntime
{
public:
    explicit NNRuntime(ConvolutionWeights weights);

    std::vector<float> infer(const std::vector<float> &input) const;

private:
    ConvolutionWeights weights;
};

} // namespace audio::firFilter
