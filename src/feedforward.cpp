#include "feedforward.h"

#include "feedforward/benchmark/BenchmarkComparison.h"
#include "feedforward/common/FeedforwardTypes.h"
#include "feedforward/runtime/LibTorchRuntime.h"
#include "feedforward/runtime/NNRuntime.h"

#include <iostream>
#include <memory>

int runFeedforwardDemo()
{
    const auto samples = feedforward::makeXorSamples();
    auto torchModel = std::make_shared<feedforward::FeedforwardNetwork>();
    feedforward::LibTorchRuntime libTorchRuntime(torchModel);
    feedforward::NNRuntime nnRuntime(feedforward::exportWeights(*torchModel));
    const auto torchInputs = libTorchRuntime.prepareInputs(samples);
    const auto runtimeInputs = nnRuntime.prepareInputs(samples);

    const auto comparisons = feedforward::compareOutputs(libTorchRuntime,
                                                         torchInputs,
                                                         nnRuntime,
                                                         runtimeInputs,
                                                         samples);

    feedforward::printComparisons(libTorchRuntime.name(),
                                  nnRuntime.name(),
                                  comparisons,
                                  std::cout);

    libTorchRuntime.runBatch(torchInputs);
    nnRuntime.runBatch(runtimeInputs);

    std::cout << "Inference benchmark" << std::endl;
    const double libTorchTime = feedforward::benchmarkMicrosecondsPerBatch(
        feedforward::benchmarkIterations,
        [&libTorchRuntime, &torchInputs] {
            return libTorchRuntime.runBatch(torchInputs);
        });
    const double runtimeTime = feedforward::benchmarkMicrosecondsPerBatch(
        feedforward::benchmarkIterations,
        [&nnRuntime, &runtimeInputs] {
            return nnRuntime.runBatch(runtimeInputs);
        });

    std::cout << libTorchRuntime.name() << ": " << libTorchTime
              << " us per XOR batch" << std::endl;
    std::cout << nnRuntime.name() << ": " << runtimeTime
              << " us per XOR batch" << std::endl;

    return 0;
}
