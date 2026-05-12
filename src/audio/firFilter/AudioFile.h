#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace audio::firFilter {

struct MonoAudio
{
    uint32_t sampleRate = 44100;
    std::vector<float> samples;
};

MonoAudio readMonoWav(const std::string &path);

void writeMonoWav(const std::string &path, const MonoAudio &audio);

} // namespace audio::firFilter
