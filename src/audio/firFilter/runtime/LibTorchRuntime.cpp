#include "LibTorchRuntime.h"

#include <stdexcept>
#include <utility>

namespace audio::firFilter {

namespace {

torch::Tensor audioTensor(const std::vector<float> &values)
{
    const auto options = torch::TensorOptions().dtype(torch::kFloat32);
    return torch::tensor(values, options)
        .reshape({1,
                  static_cast<int64_t>(inputChannels),
                  static_cast<int64_t>(values.size())});
}

std::vector<float> tensorValues(const torch::Tensor &tensor)
{
    const auto values = tensor.detach().to(torch::kCPU).contiguous();
    const auto *data = static_cast<const float *>(values.data_ptr());
    const auto count = static_cast<size_t>(values.numel());
    return {data, data + count};
}

} // namespace

LibTorchRuntime::LibTorchRuntime(
    std::shared_ptr<FIRFilterNetwork> trainedNetwork)
    : network(std::move(trainedNetwork))
{
    if (!network) {
        throw std::invalid_argument("LibTorchRuntime requires a network");
    }

    network->eval();
}

std::vector<float> LibTorchRuntime::infer(const std::vector<float> &input)
{
    torch::NoGradGuard noGrad;
    return tensorValues(network->forward(audioTensor(input)));
}

} // namespace audio::firFilter
