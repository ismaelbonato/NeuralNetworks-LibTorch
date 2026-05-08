#include "FeedforwardModel.h"

#include <cstdint>
#include <vector>

namespace feedforward {

namespace {

std::vector<float> tensorValues(const torch::Tensor &tensor)
{
    const auto values = tensor.detach().to(torch::kCPU).contiguous();
    const auto *data = static_cast<const float *>(values.data_ptr());
    const auto count = static_cast<size_t>(values.numel());
    return {data, data + count};
}

DenseLayerWeights denseWeightsFromLayer(const std::string &name,
                                        const size_t inputSize,
                                        const size_t outputSize,
                                        const torch::nn::Linear &layer)
{
    DenseLayerWeights weights;
    weights.name = name;
    weights.inputSize = inputSize;
    weights.outputSize = outputSize;
    weights.weights = tensorValues(layer->weight);
    weights.biases = tensorValues(layer->bias);
    return weights;
}

} // namespace

FeedforwardNetwork::FeedforwardNetwork()
    : inputLayer(register_module("input", torch::nn::Linear(2, 2)))
    , outputLayer(register_module("output", torch::nn::Linear(2, 1)))
{
    torch::NoGradGuard noGrad;
    const auto options = torch::TensorOptions().dtype(torch::kFloat32);

    inputLayer->weight.copy_(torch::tensor(std::vector<float>{8.051888F,
                                                              8.051895F,
                                                              -8.016418F,
                                                              -8.016412F},
                                           options)
                                 .reshape({2, 2}));
    inputLayer->bias.copy_(
        torch::tensor(std::vector<float>{-3.967814F, 12.036060F}, options));

    outputLayer->weight.copy_(
        torch::tensor(std::vector<float>{8.243962F, 8.242302F}, options)
            .reshape({1, 2}));
    outputLayer->bias.copy_(
        torch::tensor(std::vector<float>{-12.212015F}, options));
}

torch::Tensor FeedforwardNetwork::forward(const torch::Tensor &input)
{
    auto hidden = torch::sigmoid(inputLayer->forward(input));
    return torch::sigmoid(outputLayer->forward(hidden));
}

FeedforwardWeights exportWeights(const FeedforwardNetwork &network)
{
    FeedforwardWeights weights;
    weights.layers.push_back(denseWeightsFromLayer("runtime input dense layer",
                                                   2,
                                                   2,
                                                   network.inputLayer));
    weights.layers.push_back(denseWeightsFromLayer("runtime output dense layer",
                                                   2,
                                                   1,
                                                   network.outputLayer));
    return weights;
}

} // namespace feedforward
