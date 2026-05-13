#pragma once

#include "audio/firFilter/FIRFilterTypes.h"

#include <cstdint>
#include <string>
#include <string_view>

namespace audio::firFilter {

struct FIRFilterModel
{
    uint32_t sampleRate = 44100;
    ConvolutionWeights weights;
};

FIRFilterModel loadFIRFilterModelJson(std::string_view json);

FIRFilterModel loadFIRFilterModelJsonFile(const std::string &path);

} // namespace audio::firFilter
