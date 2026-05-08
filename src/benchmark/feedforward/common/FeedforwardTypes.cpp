#include "FeedforwardTypes.h"

namespace feedforward {

std::vector<InferenceSample> makeXorSamples()
{
    return {
        {{0.0F, 0.0F}},
        {{0.0F, 1.0F}},
        {{1.0F, 0.0F}},
        {{1.0F, 1.0F}},
    };
}

} // namespace feedforward
