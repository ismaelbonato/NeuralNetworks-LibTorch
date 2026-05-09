#include "LowPassFIRModel.h"
#include "src/audio/lowPassFIR/AudioFile.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace audio::lowPassFIR {

namespace {

constexpr size_t kernelRadius = (kernelSize - 1) / 2;

std::vector<float> cropSamples(const std::vector<float> &samples,
                               const size_t start,
                               const size_t length)
{
    if (start > samples.size() || length > samples.size() - start) {
        throw std::invalid_argument("crop range exceeds sample count");
    }

    const auto cropStart = samples.begin() + static_cast<std::ptrdiff_t>(start);
    const auto cropEnd = cropStart + static_cast<std::ptrdiff_t>(length);
    return {cropStart, cropEnd};
}

size_t validConvolutionOutputLength(const size_t inputLength)
{
    if (inputLength < kernelSize) {
        throw std::invalid_argument("input must be at least kernel length");
    }

    return inputLength - kernelSize + 1;
}

std::vector<float> cropTargetForValidConvolution(
    const std::vector<float> &target, const size_t inputLength)
{
    return cropSamples(target,
                       kernelRadius,
                       validConvolutionOutputLength(inputLength));
}

torch::Tensor audioTensor(const std::vector<float> &values)
{
    const auto options = torch::TensorOptions().dtype(torch::kFloat32);
    // LibTorch Conv1d expects {batch, channels, samples}.
    return torch::tensor(values, options)
        .reshape({1,
                  static_cast<int64_t>(inputChannels),
                  static_cast<int64_t>(values.size())});
}

std::vector<float> tensorValues(const torch::Tensor &tensor)
{
    // Export a contiguous CPU copy so the runtime side does not depend on LibTorch.
    const auto values = tensor.detach().to(torch::kCPU).contiguous();
    const auto *data = static_cast<const float *>(values.data_ptr());
    const auto count = static_cast<size_t>(values.numel());
    return {data, data + count};
}

float lossValue(LowPassFIRNetwork &network,
                const torch::Tensor &input,
                const torch::Tensor &target)
{
    // Evaluation loss should not allocate gradients.
    torch::NoGradGuard noGrad;
    return torch::mse_loss(network.forward(input), target).item<float>();
}

void printWeights(const char *label, const ConvolutionWeights &weights)
{
    std::cout << label << ": [";
    for (size_t index = 0; index < weights.weights.size(); ++index) {
        if (index > 0) {
            std::cout << ", ";
        }
        std::cout << weights.weights[index];
    }
    std::cout << "]" << std::endl;
}

} // namespace

LowPassFIRNetwork::LowPassFIRNetwork()
    : convolution(register_module(
          "lowPassFIR",
          // A mono Conv1d without bias is the same shape as a simple FIR filter.
          torch::nn::Conv1d(
              torch::nn::Conv1dOptions(inputChannels, outputChannels, kernelSize)
                  .stride(1)
                  .padding(0)
                  .with_bias(false))))
{}

torch::Tensor LowPassFIRNetwork::forward(const torch::Tensor &input)
{
    return convolution->forward(input);
}

TrainingFixture makeTrainingFixture()
{
    TrainingFixture fixture;
    // Input stays full length; valid convolution shortens only the prediction.
    fixture.input = audio::lowPassFIR::readMonoWav(audioSample).samples;

    const auto fullTarget = audio::lowPassFIR::readMonoWav(audioTarget).samples;
    fixture.target = cropTargetForValidConvolution(fullTarget,
                                                   fixture.input.size());
    return fixture;
}

ConvolutionWeights exportWeights(const LowPassFIRNetwork &network)
{
    ConvolutionWeights weights;
    // LibTorch Conv1d stores weights as {outputChannels, inputChannels, kernel}.
    weights.weights = tensorValues(network.convolution->weight);
    weights.biases = {0.0F};
    return weights;
}

TrainingResult trainLowPassFIR()
{
    // Keep the initial kernel deterministic while this is still a fixture.
    torch::manual_seed(7);

    TrainingResult result;
    result.network = std::make_shared<LowPassFIRNetwork>();
    result.fixture = makeTrainingFixture();

    // Convert fixture vectors to the Conv1d tensor layout.
    auto input = audioTensor(result.fixture.input);
    auto target = audioTensor(result.fixture.target);

    // Capture loss before training so callers can verify learning happened.
    result.initialLoss = lossValue(*result.network, input, target);
    printWeights("initial weights", exportWeights(*result.network));

    torch::optim::Adam optimizer(result.network->parameters(),
                                 torch::optim::AdamOptions(0.03));
    result.network->train();

    for (size_t epoch = 0; epoch < trainingEpochs; ++epoch) {
        // Clear gradients from the previous optimization step.
        optimizer.zero_grad();

        // Run the current FIR kernel over the synthetic audio input.
        auto prediction = result.network->forward(input);

        // Compare the prediction with the target low-pass filtered signal.
        auto loss = torch::mse_loss(prediction, target);

        // Compute gradients for the convolution weights from that error.
        loss.backward();

        // Update the weights using Adam.
        optimizer.step();
    }

    result.network->eval();
    result.finalLoss = lossValue(*result.network, input, target);
    result.weights = exportWeights(*result.network);
    printWeights("final weights", result.weights);
    return result;
}

} // namespace audio::lowPassFIR
