# Feedforward Runtime Benchmark Example

This project demonstrates a small **feedforward neural network** implemented in C++ using [LibTorch](https://pytorch.org/cppdocs/) and then mirrored into `nn-runtime` through app-local exported weights.

## Features

- Defines a two-layer network with a `2 -> 2` hidden layer and a `2 -> 1` output layer.
- Loads deterministic XOR fixture weights into LibTorch `Linear` layers.
- Exports those weights into small in-memory structs in the app layer.
- Builds the equivalent `nn-runtime` model without adding LibTorch awareness to `nn-runtime`.
- Prints predictions for the four XOR inputs from both backends.
- Benchmarks inference after both backends and input batches have already been constructed.

## Requirements

- [LibTorch](https://pytorch.org/get-started/locally/) (tested with 1.13+)
- C++17 compiler

## Usage

1. **Build with CMake:**  
    ```sh
    cmake --preset clang-debug
    cmake --build --preset clang-debug
    ```

2. **Run:**
   ```
   ./build-clang/nn-training
   ```

## How it Works

- **Network Construction:**  
  The app registers two LibTorch `Linear` layers: one input layer with 2 inputs and 2 outputs, and one output layer with 2 inputs and 1 output.

- **Weight Export:**  
  The app exports each LibTorch dense layer into local `DenseLayerWeights` / `FeedforwardWeights` structs. These structs stay outside `libs/nn-runtime`.

- **Runtime Mirror:**  
  The app converts the exported weights into `nn::DenseLayerRecipe` and `nn::Parameters`, then builds an equivalent `nn-runtime` model.

- **Output:**  
  The executable prints LibTorch and `nn-runtime` predictions for each XOR input so parity can be checked before benchmarking.

- **Benchmarking:**  
  The LibTorch model, `nn-runtime` model, and input batches are all created before timing starts. The benchmark loop measures repeated inference batches only.
