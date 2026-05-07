#include <filesystem>
#include <iostream>
#include <torch/torch.h>
#include <vector>

#include <opencv2/opencv.hpp>

std::vector<double> png_to_bits(const std::string &filename)
{
    cv::Mat img = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    cv::Mat resizedImage;

    cv::resize(img, resizedImage, cv::Size(64, 64));

    std::vector<double> pattern;
    pattern.reserve(static_cast<size_t>(resizedImage.rows)
                    * static_cast<size_t>(resizedImage.cols));

    for (size_t r = 0; r < static_cast<size_t>(resizedImage.rows); ++r) {
        for (size_t c = 0; c < static_cast<size_t>(resizedImage.cols); ++c) {
            auto pixel = resizedImage.at<uchar>(static_cast<int>(r),
                                                static_cast<int>(c));
            pattern.emplace_back(pixel > 128 ? 1 : -1);
        }
    }
    return pattern;
}

// Helper: sign function for tensors
torch::Tensor custom_sign(torch::Tensor x)
{
    return torch::where(x >= 0, torch::ones_like(x), -torch::ones_like(x));
}

int runHopfieldDemo()
{
    // Define patterns to store (binary: -1, 1)
    std::vector<torch::Tensor> patterns = {
        torch::tensor(png_to_bits("Misc/bart.png"), torch::kFloat32),
        torch::tensor(png_to_bits("Misc/homer.png"), torch::kFloat32),
        torch::tensor(png_to_bits("Misc/marge.png"), torch::kFloat32),
        //torch::tensor(png_to_bits("Misc/meg.png"), torch::kFloat32),
        //torch::tensor(png_to_bits("Misc/grandpa.png"), torch::kFloat32),
        //torch::tensor(png_to_bits("Misc/lisa.png"), torch::kFloat32),
        //torch::tensor(png_to_bits("Misc/mrburns.png"), torch::kFloat32),
    };

    const int64_t N = patterns[0].size(0);
    torch::Tensor W = torch::zeros({N, N});

    // Hebbian learning rule: W = sum(p * p^T), zero diagonal
    for (const auto &p : patterns) {
        W += torch::ger(p, p);
    }
    W.fill_diagonal_(0);

    // Test recall with a noisy pattern
    torch::Tensor test = torch::tensor(png_to_bits("Misc/homer_defect.png"),
                                       torch::kFloat32);

    std::cout << "Initial (noisy) input: " << std::endl;
    size_t idx = 0;
    for (int64_t i = 0; i < test.size(0); ++i) {
        if (idx == 64) {
            std::cout << std::endl;
            idx = 0;
        }
        const float value = test[i].item<float>();
        std::cout << (value > 0 ? "o" : " ");
        idx++;
    }
    std::cout << std::endl;

    // Update until convergence (asynchronous update for simplicity)
    torch::Tensor prev;
    int steps = 0;
    do {
        prev = test.clone();
        test = custom_sign(torch::matmul(W, test));
        steps++;
    } while (!torch::allclose(test, prev) && steps < 10);

    std::cout << "Recalled pattern: " << std::endl;

    idx = 0;
    for (int64_t i = 0; i < test.size(0); ++i) {
        if (idx == 64) {
            std::cout << std::endl;
            idx = 0;
        }
        const float value = test[i].item<float>();
        std::cout << (value > 0 ? "o" : " ");
        idx++;
    }
    std::cout << std::endl;

    return 0;
}
