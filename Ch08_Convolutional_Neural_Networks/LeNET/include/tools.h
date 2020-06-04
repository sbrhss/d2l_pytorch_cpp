

#pragma once

#include <torch/torch.h>


class Reshape : public torch::nn::Module{
public:
    torch::Tensor forward(const torch::Tensor& input){
            return input.view({-1,1,28,28});
    }
};

class Flatten : public torch::nn::Module{
public:
    torch::Tensor forward(const torch::Tensor& input){
            return input.view({input.sizes()[0], -1});
    }
};

class LeNET : public torch::nn::Module{
public:
   explicit LeNET():
   Network_(
      Reshape(),
      torch::nn::Conv2d(torch::nn::Conv2dOptions(/*in_channels=*/1, /*out_channels=*/6, /*kernel_size=*/5).padding(/*padding=*/2).bias(false)),
      torch::nn::Sigmoid(),
      torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(/*kernel_size=*/2).stride(2)),
       torch::nn::Conv2d(torch::nn::Conv2dOptions(/*in_channels=*/6, /*out_channels=*/16, /*kernel_size=*/5)),
       torch::nn::Sigmoid(),
       torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(/*kernel_size=*/2).stride(2)),
       Flatten(),
       torch::nn::Linear(/*in_features=*/16*5*5, /*out_features=*/120),
       torch::nn::Sigmoid(),
       torch::nn::Linear(120, 84),
       torch::nn::Sigmoid(),
       torch::nn::Linear(84, 10)
      )
   {
       
       
       register_module("Network_", Network_);
   }
   torch::Tensor forward(torch::Tensor& x){
      return Network_->forward(x);
   }

private:
   torch::nn::Sequential Network_;

};


float accuracy (const torch::Tensor& y_hat,
                const torch::Tensor& y);

template <typename DataLoader>
float train(
   LeNET& model,
   size_t epoch,
   DataLoader& data_loader,
   torch::optim::Optimizer& optimizer,
   size_t kNumInputs,
   const size_t dataset_size );


template <typename DataLoader>
float test(
   LeNET& model,
   size_t epoch,
   DataLoader& data_loader,
   torch::optim::Optimizer& optimizer,
   size_t kNumInputs,
   const size_t dataset_size );
