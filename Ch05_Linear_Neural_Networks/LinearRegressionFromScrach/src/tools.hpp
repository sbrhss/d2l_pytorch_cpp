#pragma once

#include <torch/torch.h>
#include <torch/script.h>
#include <algorithm>    // std::random_shuffle
#include <vector>       // std::vector
#include "../include/matplotlibcpp.h"


void data_iter(const int batch_size, const torch::Tensor& features, const torch::Tensor& labels, std::vector<torch::Tensor>& vector_features, std::vector<torch::Tensor>& vactor_labels);
torch::Tensor linreg(const torch::Tensor& X, const torch::Tensor& w, const torch::Tensor& b);
template <typename T>
std::vector<T> linspace(T a, T b, size_t N);
void sgd(const torch::Tensor& w, const torch::Tensor& b, const float lr, const size_t batch_size) ;
torch::Tensor squared_loss(torch::Tensor& y_hat, torch::Tensor& y);