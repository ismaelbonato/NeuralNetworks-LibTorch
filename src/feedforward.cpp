#include "feedforward.h"

#include "base/ActivationFunction.h"
#include "base/Model.h"
#include "layers/DenseLayer.h"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <torch/torch.h>
#include <vector>

namespace {

constexpr size_t benchmarkIterations = 100000;

struct DenseLayerWeights
{
    std::string name;
    size_t inputSize = 0;
    size_t outputSize = 0;
    std::vector<float> weights;
    std::vector<float> biases;
};

struct FeedforwardWeights
{
    DenseLayerWeights inputLayer;
    DenseLayerWeights outputLayer;
};

struct FeedforwardNetwork : torch::nn::Module
{
    FeedforwardNetwork()
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

    torch::Tensor forward(const torch::Tensor &input)
    {
        auto hidden = torch::sigmoid(inputLayer->forward(input));
        return torch::sigmoid(outputLayer->forward(hidden));
    }

    torch::nn::Linear inputLayer{nullptr};
    torch::nn::Linear outputLayer{nullptr};
};

std::vector<float> tensorValues(const torch::Tensor &tensor)
{
    const auto values = tensor.detach().to(torch::kCPU).contiguous();
    const auto *data = static_cast<const float *>(values.data_ptr());
    const auto count = static_cast<size_t>(values.numel());
    return {data, data + count};
}

FeedforwardWeights exportWeights(const FeedforwardNetwork &network)
{
    FeedforwardWeights weights;
    weights.inputLayer.name = "runtime input dense layer";
    weights.inputLayer.inputSize = 2;
    weights.inputLayer.outputSize = 2;
    weights.inputLayer.weights = tensorValues(network.inputLayer->weight);
    weights.inputLayer.biases = tensorValues(network.inputLayer->bias);

    weights.outputLayer.name = "runtime output dense layer";
    weights.outputLayer.inputSize = 2;
    weights.outputLayer.outputSize = 1;
    weights.outputLayer.weights = tensorValues(network.outputLayer->weight);
    weights.outputLayer.biases = tensorValues(network.outputLayer->bias);

    return weights;
}

nn::Parameters toRuntimeParameters(const DenseLayerWeights &weights)
{
    nn::Pattern runtimeWeights(weights.weights.begin(), weights.weights.end());
    runtimeWeights.reshape({weights.outputSize, weights.inputSize});

    nn::Parameters parameters;
    parameters.weights = runtimeWeights;
    parameters.biases = nn::Pattern(weights.biases.begin(),
                                    weights.biases.end());
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

struct RuntimeFeedforwardNetwork
{
    explicit RuntimeFeedforwardNetwork(const FeedforwardWeights &weights)
    {
        model.addLayer(makeRuntimeDenseLayer(weights.inputLayer));
        model.addLayer(makeRuntimeDenseLayer(weights.outputLayer));
    }

    nn::Pattern infer(const nn::Pattern &input)
    {
        return model.infer(input);
    }

    nn::Model model;
};

std::vector<torch::Tensor> makeTorchInputs()
{
    return {
        torch::tensor(std::vector<float>{0.0F, 0.0F}),
        torch::tensor(std::vector<float>{0.0F, 1.0F}),
        torch::tensor(std::vector<float>{1.0F, 0.0F}),
        torch::tensor(std::vector<float>{1.0F, 1.0F}),
    };
}

std::vector<nn::Pattern> makeRuntimeInputs()
{
    return {
        nn::Pattern{0.0F, 0.0F},
        nn::Pattern{0.0F, 1.0F},
        nn::Pattern{1.0F, 0.0F},
        nn::Pattern{1.0F, 1.0F},
    };
}

float runLibTorchBatch(FeedforwardNetwork &network,
                       const std::vector<torch::Tensor> &inputs)
{
    torch::NoGradGuard noGrad;
    float total = 0.0F;
    for (const auto &input : inputs) {
        total += network.forward(input).item<float>();
    }
    return total;
}

float runRuntimeBatch(RuntimeFeedforwardNetwork &network,
                      const std::vector<nn::Pattern> &inputs)
{
    float total = 0.0F;
    for (const auto &input : inputs) {
        total += network.infer(input)[0];
    }
    return total;
}

template<typename Operation>
double benchmarkMicrosecondsPerBatch(Operation operation)
{
    volatile float sink = 0.0F;
    const auto start = std::chrono::steady_clock::now();
    for (size_t iteration = 0; iteration < benchmarkIterations; ++iteration) {
        sink += operation();
    }
    const auto end = std::chrono::steady_clock::now();

    const auto elapsed =
        std::chrono::duration<double, std::micro>(end - start).count();
    return elapsed / static_cast<double>(benchmarkIterations);
}

void printBenchmarkResult(const char *name, const double microsecondsPerBatch)
{
    std::cout << name << ": " << microsecondsPerBatch
              << " us per XOR batch" << std::endl;
}

} // namespace

int runFeedforwardDemo()
{
    FeedforwardNetwork network;
    network.eval();
    const FeedforwardWeights weights = exportWeights(network);
    RuntimeFeedforwardNetwork runtimeNetwork(weights);
    const std::vector<torch::Tensor> torchInputs = makeTorchInputs();
    const std::vector<nn::Pattern> runtimeInputs = makeRuntimeInputs();

    std::cout << "LibTorch XOR feedforward fixture" << std::endl;
    // Runtime inference using the feedforward network from LibTorch.
    for (const auto &input : torchInputs) {
        const auto prediction = network.forward(input);
        std::cout << '[' << input[0].item<float>() << ", "
                  << input[1].item<float>() << "] -> "
                  << prediction.item<float>() << std::endl;
    }

    std::cout << "nn-runtime XOR feedforward fixture" << std::endl;
    for (const auto &input : runtimeInputs) {
        const auto prediction = runtimeNetwork.infer(input);
        std::cout << '[' << input[0] << ", " << input[1] << "] -> "
                  << prediction[0] << std::endl;
    }

    runLibTorchBatch(network, torchInputs);
    runRuntimeBatch(runtimeNetwork, runtimeInputs);

    std::cout << "Inference benchmark" << std::endl;
    printBenchmarkResult(
        "LibTorch",
        benchmarkMicrosecondsPerBatch(
            [&network, &torchInputs] {
                return runLibTorchBatch(network, torchInputs);
            }));
    printBenchmarkResult(
        "nn-runtime",
        benchmarkMicrosecondsPerBatch(
            [&runtimeNetwork, &runtimeInputs] {
                return runRuntimeBatch(runtimeNetwork, runtimeInputs);
            }));

    return 0;
}
