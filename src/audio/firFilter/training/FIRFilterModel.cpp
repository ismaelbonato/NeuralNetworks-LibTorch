#include "FIRFilterModel.h"
#include "src/audio/firFilter/AudioFile.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace audio::firFilter {

namespace {

size_t validConvolutionOutputLength(const size_t inputLength,
                                    const size_t kernelSize)
{
    if (inputLength < kernelSize) {
        throw std::invalid_argument("input must be at least kernel length");
    }

    return inputLength - kernelSize + 1;
}

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

std::vector<float> cropAlignedTarget(const std::vector<float> &target,
                                     const size_t inputLength,
                                     const FIRFilterConfig &config)
{
    return cropSamples(target,
                       config.targetAlignmentOffset,
                       validConvolutionOutputLength(inputLength,
                                                    config.kernelSize));
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

float lossValue(FIRFilterNetwork &network,
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

void writeFloatArray(std::ostream &output, const std::vector<float> &values)
{
    output << "[";
    for (size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            output << ", ";
        }
        output << std::setprecision(9) << values[index];
    }
    output << "]";
}

} // namespace

FIRFilterNetwork::FIRFilterNetwork(const FIRFilterConfig &config)
    : convolution(register_module(
          "firFilter",
          // A mono Conv1d without bias is the same shape as a simple FIR filter.
          torch::nn::Conv1d(
              torch::nn::Conv1dOptions(inputChannels,
                                        outputChannels,
                                        static_cast<int64_t>(config.kernelSize))
                  .stride(1)
                  .padding(0)
                  .with_bias(false))))
{}

torch::Tensor FIRFilterNetwork::forward(const torch::Tensor &input)
{
    return convolution->forward(input);
}

TrainingFixture loadTrainingFixture(const FIRFilterConfig &config)
{
    TrainingFixture fixture;
    // Input stays full length; valid convolution shortens only the prediction.
    const auto inputAudio = audio::firFilter::readMonoWav(
        config.trainingInputPath);
    const auto targetAudio = audio::firFilter::readMonoWav(
        config.trainingTargetPath);

    fixture.input = inputAudio.samples;
    fixture.sampleRate = inputAudio.sampleRate;
    fixture.target = cropAlignedTarget(targetAudio.samples,
                                       fixture.input.size(),
                                       config);
    return fixture;
}

ConvolutionWeights exportWeights(const FIRFilterNetwork &network,
                                 const FIRFilterConfig &config,
                                 const size_t inputLength)
{
    ConvolutionWeights weights;
    // LibTorch Conv1d stores weights as {outputChannels, inputChannels, kernel}.
    weights.inputLength = inputLength;
    weights.kernelLength = config.kernelSize;
    weights.weights = tensorValues(network.convolution->weight);
    weights.biases = {0.0F};
    return weights;
}

TrainingResult trainFIRFilter(const FIRFilterConfig &config)
{
    // Keep the initial kernel deterministic while this is still a fixture.
    torch::manual_seed(7);

    TrainingResult result;
    result.network = std::make_shared<FIRFilterNetwork>(config);
    result.fixture = loadTrainingFixture(config);

    // Convert fixture vectors to the Conv1d tensor layout.
    auto input = audioTensor(result.fixture.input);
    auto target = audioTensor(result.fixture.target);

    // Capture loss before training so callers can verify learning happened.
    result.initialLoss = lossValue(*result.network, input, target);
    printWeights("initial weights",
                 exportWeights(*result.network,
                               config,
                               result.fixture.input.size()));

    torch::optim::Adam optimizer(result.network->parameters(),
                                 torch::optim::AdamOptions(0.03));
    result.network->train();

    for (size_t epoch = 0; epoch < config.trainingEpochs; ++epoch) {
        // Clear gradients from the previous optimization step.
        optimizer.zero_grad();

        // Run the current FIR kernel over the training audio input.
        auto prediction = result.network->forward(input);

        // Compare the prediction with the target filtered signal.
        auto loss = torch::mse_loss(prediction, target);

        // Compute gradients for the convolution weights from that error.
        loss.backward();

        // Update the weights using Adam.
        optimizer.step();
    }

    result.network->eval();
    result.finalLoss = lossValue(*result.network, input, target);
    result.weights = exportWeights(*result.network,
                                   config,
                                   result.fixture.input.size());
    printWeights("final weights", result.weights);
    return result;
}

void writeModelJson(const std::string &path,
                    const ConvolutionWeights &weights,
                    const uint32_t sampleRate)
{
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Could not write FIR filter model: " + path);
    }

    output << "{\n";
    output << "  \"format\": \"nn-training.fir-filter.v1\",\n";
    output << "  \"sampleRate\": " << sampleRate << ",\n";
    output << "  \"architecture\": {\n";
    output << "    \"type\": \"Conv1D\",\n";
    output << "    \"inputChannels\": " << weights.inputChannelCount << ",\n";
    output << "    \"outputChannels\": " << weights.outputChannelCount << ",\n";
    output << "    \"kernelSize\": " << weights.kernelLength << ",\n";
    output << "    \"stride\": " << weights.stride << ",\n";
    output << "    \"padding\": " << weights.padding << ",\n";
    output << "    \"activation\": \"identity\"\n";
    output << "  },\n";
    output << "  \"parameters\": {\n";
    output << "    \"weights\": ";
    writeFloatArray(output, weights.weights);
    output << ",\n";
    output << "    \"biases\": ";
    writeFloatArray(output, weights.biases);
    output << "\n";
    output << "  }\n";
    output << "}\n";
}

} // namespace audio::firFilter
