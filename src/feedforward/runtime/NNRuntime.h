#pragma once

#include "feedforward/common/FeedforwardTypes.h"

#include "base/Model.h"
#include "base/Types.h"

#include <vector>

namespace feedforward {

class NNRuntime
{
public:
    using InputBatch = std::vector<nn::Pattern>;

    explicit NNRuntime(const FeedforwardWeights &weights);

    const char *name() const;
    InputBatch prepareInputs(const std::vector<InferenceSample> &samples) const;
    float infer(const nn::Pattern &input);
    float runBatch(const InputBatch &inputs);

private:
    nn::Model model;
};

} // namespace feedforward
