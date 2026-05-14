# FIR filter training pipeline

The FIR filter path learns a fixed linear filter from paired WAV files:

```text
training input WAV -> processed target WAV
```

The input and target must be the same underlying signal. For example, white
noise and a filtered copy of that same white noise are a valid pair. Two
different recordings that merely sound similar are not a valid waveform target.

## Alignment contract

The model uses a valid 1D convolution with no padding. For an input with `N`
samples and a kernel with `K` taps:

```text
output length = N - K + 1
```

For a centered, zero-phase target, crop the target by the kernel radius:

```text
targetAlignmentOffset = (kernelSize - 1) / 2
```

With the current default `kernelSize = 101`, the centered offset is `50`.
The first prediction uses `input[0..100]` and is compared to `target[50]`.

If the target was created by a causal filter, hardware, Audacity, a pedal, or
another external process, it may contain additional latency. In that case,
measure the delay and use that measured value as `targetAlignmentOffset`
instead of the centered kernel radius.

## Kernel size behavior

A larger kernel can represent sharper or longer linear filters, but it also
shortens valid-convolution output by more samples:

```text
lost samples = kernelSize - 1
```

For example:

```text
kernelSize = 101 -> output loses 100 samples
kernelSize = 501 -> output loses 500 samples
```

The training WAV only needs to be longer than the kernel, but longer training
audio gives the optimizer more examples of the same filter behavior.

## Generated files

Running `nn-training` keeps the flow in one command:

```text
train FIR model
export JSON
load JSON
run LibTorch runtime
run in-memory nn-runtime
run JSON-loaded nn-runtime
write outputs
print verification summary
```

Generated artifacts are written relative to the process working directory:

```text
output/fir-filter/libtorch-output.wav
output/fir-filter/nn-runtime-output.wav
output/fir-filter/fir-filter-model.json
```

The JSON file is the deployment bridge for `nn-runtime`. LibTorch remains a
training and validation dependency only.
