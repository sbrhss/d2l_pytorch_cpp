

#pragma once

#include <torch/torch.h>

const int64_t kNumHiddenSize = 200;
const size_t kNumOutputs = 10;
const size_t kNumInputs = 784;


class Net : public torch::nn::Module {
public:
   explicit Net()
      :
      fc1(kNumInputs, kNumHiddenSize),
      fc2(kNumHiddenSize, kNumOutputs) {


      register_module("fc1", fc1);
      register_module("fc2", fc2);

   }

   torch::Tensor forward(torch::Tensor& x) {
      x = x.view({ -1, 784});
      x = torch::relu(fc1->forward(x));
      x = fc2->forward(x);
      return torch::log_softmax(x, /*dim=*/1);
      //return x;
   }

private:
   torch::nn::Linear fc1;
   torch::nn::Linear fc2;

};


float accuracy (const torch::Tensor& y_hat,
                const torch::Tensor& y);

template <typename DataLoader>
float train(
   Net& model,
   size_t epoch,
   DataLoader& data_loader,
   torch::optim::Optimizer& optimizer,
   size_t kNumInputs,
   const size_t dataset_size );


template <typename DataLoader>
float test(
   Net& model,
   size_t epoch,
   DataLoader& data_loader,
   torch::optim::Optimizer& optimizer,
   size_t kNumInputs,
   const size_t dataset_size );
