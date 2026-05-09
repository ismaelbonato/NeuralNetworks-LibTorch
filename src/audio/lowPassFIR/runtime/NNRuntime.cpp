#include "NNRuntime.h"

#include "base/ActivationFunction.h"
#include "base/Model.h"
#include "base/Parameters.h"
#include "base/Types.h"
#include "layers/ConvolutionalLayer.h"

#include <memory>
#include <stdexcept>
#include <utility>

namespace audio::lowPassFIR {

namespace {

nn::Pattern runtimeInput(const std::vector<float> &input)
{
    nn::Pattern pattern(input.begin(), input.end());
    pattern.reshape({inputChannels, input.size()});
    return pattern;
}

nn::Parameters toRuntimeParameters(const ConvolutionWeights &weights)
{
    nn::Pattern runtimeWeights(weights.weights.begin(), weights.weights.end());
    runtimeWeights.reshape({weights.outputChannelCount,
                            weights.inputChannelCount,
                            weights.kernelLength});

    nn::Parameters parameters;
    parameters.weights = runtimeWeights;
    parameters.biases = nn::Pattern(weights.biases.begin(), weights.biases.end());
    return parameters;
}

std::unique_ptr<nn::ConvolutionalLayer> makeRuntimeConvolutionalLayer(
    const ConvolutionWeights &weights,
    const size_t inputLength)
{
    nn::ConvolutionalLayerRecipe recipe;
    recipe.name = "low-pass FIR";
    recipe.type = "ConvolutionalLayer";
    recipe.info = "exported LibTorch low-pass FIR layer";
    recipe.activation = std::make_shared<nn::IdentityActivation<nn::Scalar>>();
    recipe.inputChannels = weights.inputChannelCount;
    recipe.inputLength = inputLength;
    recipe.outputChannels = weights.outputChannelCount;
    recipe.kernelSize = weights.kernelLength;
    recipe.stride = weights.stride;
    recipe.padding = weights.padding;

    auto layer = std::make_unique<nn::ConvolutionalLayer>(recipe);
    layer->setParameters(toRuntimeParameters(weights));
    return layer;
}

std::vector<float> flattenedValues(const nn::Pattern &pattern)
{
    return {pattern.begin(), pattern.end()};
}

} // namespace

NNRuntime::NNRuntime(ConvolutionWeights newWeights)
    : weights(std::move(newWeights))
{
    if (weights.weights.empty()) {
        throw std::invalid_argument("NNRuntime requires FIR weights");
    }
}

std::vector<float> NNRuntime::infer(const std::vector<float> &input) const
{
    nn::Model model;
    model.addLayer(makeRuntimeConvolutionalLayer(weights, input.size()));
    return flattenedValues(model.infer(runtimeInput(input)));
}

} // namespace audio::lowPassFIR
