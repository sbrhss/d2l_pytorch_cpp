#pragma once

#include <torch/torch.h>

torch::Tensor net(const torch::Tensor& X,
                  const torch::Tensor& W,
                  const torch::Tensor& b,
                  const int kNumInputs);

torch::Tensor cross_entropy(const torch::Tensor& y_hat,
                            const torch::Tensor& y);


float accuracy (const torch::Tensor& y_hat,
                const torch::Tensor& y);


void sgd(const torch::Tensor& w,
         const torch::Tensor& b,
         const float lr,
         const size_t batch_size);


torch::Tensor softmax(const torch::Tensor& X);

template <typename DataLoader>
float train(
   const torch::Tensor& W,
   size_t epoch,
   DataLoader& data_loader,
   size_t kNumInputs,
   const torch::Tensor& b,
   const float lr,
   const size_t dataset_size );


template <typename DataLoader>
float test(
   const torch::Tensor& W,
   size_t epoch,
   DataLoader& data_loader,
   size_t kNumInputs,
   const torch::Tensor& b,
   const float lr,
   const size_t dataset_size );
