#pragma once

#include "audio/firFilter/training/FIRFilterModel.h"

#include <memory>
#include <vector>

namespace audio::firFilter {

class LibTorchRuntime
{
public:
    explicit LibTorchRuntime(std::shared_ptr<FIRFilterNetwork> trainedNetwork);

    std::vector<float> infer(const std::vector<float> &input);

private:
    std::shared_ptr<FIRFilterNetwork> network;
};

} // namespace audio::firFilter
