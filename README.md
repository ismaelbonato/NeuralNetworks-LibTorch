# Hopfield Network Example with LibTorch and OpenCV

This project demonstrates a simple **discrete Hopfield network** implemented in C++ using [LibTorch](https://pytorch.org/cppdocs/) (the C++ API for PyTorch) and [OpenCV](https://opencv.org/) for image preprocessing.

## Features

- Stores multiple binary patterns (from PNG images) in a Hopfield network.
- Recalls a pattern from a noisy version using Hebbian learning.
- Visualizes patterns as ASCII art in the terminal.

## Requirements

- [LibTorch](https://pytorch.org/get-started/locally/) (tested with 1.13+)
- [OpenCV](https://opencv.org/) (tested with 4.x)
- C++17 compiler

## Usage

1. **Prepare Images:**  
   Place your 64x64 grayscale PNG images in the appropriate directory (see the hardcoded paths in `hopfield.cpp`).
2. **Build with CMake:**  
    Then, build the project:  
    ```sh
    mkdir build
    cd build
    cmake ..
    make -j$(nproc)
    ```

3. **Run:**
   ```
   ./hopfield
   ```

## How it Works

- **Image Preprocessing:**  
  Each PNG is loaded, resized to 64x64, and binarized (`>128` becomes `1`, else `-1`).

- **Network Construction:**  
  Patterns are stored using the Hebbian rule:  
  \( W = \sum_{p} p \cdot p^T \), with zero diagonal.

- **Recall:**  
  A noisy pattern is iteratively updated until convergence using the sign activation.

- **Output:**  
  Both the noisy and recalled patterns are printed as ASCII art.