#pragma once

#include "benchmark/feedforward/common/FeedforwardTypes.h"
#include "benchmark/feedforward/runtime/LibTorchRuntime.h"
#include "benchmark/feedforward/runtime/NNRuntime.h"

#include <chrono>
#include <iosfwd>
#include <vector>

namespace feedforward {

std::vector<OutputComparison> compareOutputs(
    LibTorchRuntime &expectedRuntime,
    const LibTorchRuntime::InputBatch &expectedInputs,
    NNRuntime &actualRuntime,
    const NNRuntime::InputBatch &actualInputs,
    const std::vector<InferenceSample> &samples,
    float tolerance = outputTolerance);

bool allWithinTolerance(const std::vector<OutputComparison> &comparisons);

void printComparisons(const char *expectedName,
                      const char *actualName,
                      const std::vector<OutputComparison> &comparisons,
                      std::ostream &output);

template<typename Operation>
double benchmarkMicrosecondsPerBatch(const size_t iterations,
                                     Operation operation)
{
    volatile float sink = 0.0F;
    const auto start = std::chrono::steady_clock::now();
    for (size_t iteration = 0; iteration < iterations; ++iteration) {
        sink += operation();
    }
    const auto end = std::chrono::steady_clock::now();

    const auto elapsed =
        std::chrono::duration<double, std::micro>(end - start).count();
    return elapsed / static_cast<double>(iterations);
}

} // namespace feedforward
