#include "BenchmarkComparison.h"

#include <cmath>
#include <iostream>

namespace feedforward {

std::vector<OutputComparison> compareOutputs(
    LibTorchRuntime &expectedRuntime,
    const LibTorchRuntime::InputBatch &expectedInputs,
    NNRuntime &actualRuntime,
    const NNRuntime::InputBatch &actualInputs,
    const std::vector<InferenceSample> &samples,
    const float tolerance)
{
    std::vector<OutputComparison> comparisons;
    comparisons.reserve(samples.size());

    for (size_t index = 0; index < samples.size(); ++index) {
        OutputComparison comparison;
        comparison.input = samples[index].input;
        comparison.expected = expectedRuntime.infer(expectedInputs.at(index));
        comparison.actual = actualRuntime.infer(actualInputs.at(index));
        comparison.difference = std::fabs(comparison.expected - comparison.actual);
        comparison.withinTolerance = comparison.difference <= tolerance;
        comparisons.push_back(comparison);
    }

    return comparisons;
}

bool allWithinTolerance(const std::vector<OutputComparison> &comparisons)
{
    for (const auto &comparison : comparisons) {
        if (!comparison.withinTolerance) {
            return false;
        }
    }
    return true;
}

void printComparisons(const char *expectedName,
                      const char *actualName,
                      const std::vector<OutputComparison> &comparisons,
                      std::ostream &output)
{
    output << expectedName << " vs " << actualName << " parity" << std::endl;
    for (const auto &comparison : comparisons) {
        output << '[' << comparison.input.at(0) << ", " << comparison.input.at(1)
               << "] -> " << comparison.expected << " / " << comparison.actual
               << " diff=" << comparison.difference << std::endl;
    }
}

} // namespace feedforward
