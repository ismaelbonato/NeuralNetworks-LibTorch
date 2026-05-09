#include "AudioFile.h"

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wconversion"
#pragma clang diagnostic ignored "-Wimplicit-int-conversion"
#pragma clang diagnostic ignored "-Wimplicit-int-float-conversion"
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wsign-conversion"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#endif

#define DR_WAV_IMPLEMENTATION
#include "third_party/dr_wav/dr_wav.h"

#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include <algorithm>
#include <stdexcept>

namespace audio::lowPassFIR {

namespace {

std::string errorMessage(const std::string &action, const std::string &path)
{
    return "Could not " + action + " WAV file: " + path;
}

} // namespace

MonoAudio readMonoWav(const std::string &path)
{
    unsigned int channels = 0;
    unsigned int sampleRate = 0;
    drwav_uint64 frameCount = 0;

    float *interleavedSamples
        = drwav_open_file_and_read_pcm_frames_f32(path.c_str(),
                                                  &channels,
                                                  &sampleRate,
                                                  &frameCount,
                                                  nullptr);

    if (interleavedSamples == nullptr) {
        throw std::runtime_error(errorMessage("read", path));
    }

    if (channels == 0) {
        drwav_free(interleavedSamples, nullptr);
        throw std::runtime_error("WAV file has no channels: " + path);
    }

    MonoAudio audio;
    audio.sampleRate = sampleRate;
    audio.samples.reserve(static_cast<size_t>(frameCount));

    for (drwav_uint64 frame = 0; frame < frameCount; ++frame) {
        float monoSample = 0.0F;
        for (unsigned int channel = 0; channel < channels; ++channel) {
            monoSample += interleavedSamples[(frame * channels) + channel];
        }
        audio.samples.push_back(monoSample / static_cast<float>(channels));
    }

    drwav_free(interleavedSamples, nullptr);
    return audio;
}

void writeMonoWav(const std::string &path, const MonoAudio &audio)
{
    drwav_data_format format{};
    format.container = drwav_container_riff;
    format.format = DR_WAVE_FORMAT_IEEE_FLOAT;
    format.channels = 1;
    format.sampleRate = audio.sampleRate;
    format.bitsPerSample = 32;

    drwav wav{};
    if (!drwav_init_file_write(&wav, path.c_str(), &format, nullptr)) {
        throw std::runtime_error(errorMessage("write", path));
    }

    std::vector<float> samples = audio.samples;
    for (auto &sample : samples) {
        sample = std::clamp(sample, -1.0F, 1.0F);
    }

    const drwav_uint64 framesWritten = drwav_write_pcm_frames(&wav,
                                                              samples.size(),
                                                              samples.data());

    drwav_uninit(&wav);

    if (framesWritten != samples.size()) {
        throw std::runtime_error(errorMessage("finish writing", path));
    }
}

} // namespace audio::lowPassFIR
