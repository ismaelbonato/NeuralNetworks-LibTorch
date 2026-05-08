#include "NNRuntime.h"

#include "base/ActivationFunction.h"
#include "base/Parameters.h"
#include "layers/DenseLayer.h"

#include <memory>
#include <stdexcept>

namespace feedforward {

namespace {

nn::Parameters toRuntimeParameters(const DenseLayerWeights &weights)
{
    nn::Pattern runtimeWeights(weights.weights.begin(), weights.weights.end());
    runtimeWeights.reshape({weights.outputSize, weights.inputSize});

    nn::Parameters parameters;
    parameters.weights = runtimeWeights;
    parameters.biases = nn::Pattern(weights.biases.begin(), weights.biases.end());
    return parameters;
}

std::unique_ptr<nn::DenseLayer> makeRuntimeDenseLayer(
    const DenseLayerWeights &weights)
{
    nn::DenseLayerRecipe recipe;
    recipe.name = weights.name;
    recipe.type = "DenseLayer";
    recipe.info = "exported LibTorch XOR fixture layer";
    recipe.activation = std::make_shared<nn::SigmoidActivation<nn::Scalar>>();
    recipe.inputSize = weights.inputSize;
    recipe.outputSize = weights.outputSize;

    auto layer = std::make_unique<nn::DenseLayer>(recipe);
    layer->setParameters(toRuntimeParameters(weights));
    return layer;
}

} // namespace

NNRuntime::NNRuntime(const FeedforwardWeights &weights)
{
    if (weights.layers.empty()) {
        throw std::invalid_argument("NNRuntime requires at least one layer");
    }

    for (const auto &layer : weights.layers) {
        model.addLayer(makeRuntimeDenseLayer(layer));
    }
}

const char *NNRuntime::name() const
{
    return "nn-runtime";
}

NNRuntime::InputBatch NNRuntime::prepareInputs(
    const std::vector<InferenceSample> &samples) const
{
    InputBatch inputs;
    inputs.reserve(samples.size());
    for (const auto &sample : samples) {
        inputs.emplace_back(sample.input.begin(), sample.input.end());
    }
    return inputs;
}

float NNRuntime::infer(const nn::Pattern &input)
{
    return model.infer(input)[0];
}

float NNRuntime::runBatch(const InputBatch &inputs)
{
    float total = 0.0F;
    for (const auto &input : inputs) {
        total += infer(input);
    }
    return total;
}

} // namespace feedforward
