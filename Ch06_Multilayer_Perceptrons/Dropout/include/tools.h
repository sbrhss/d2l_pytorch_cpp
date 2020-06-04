

#pragma once

#include <torch/torch.h>

class Net : public torch::nn::Module {
public:
   explicit Net(const int64_t kNumInputs = 784, const int64_t kNumOutputs = 10, const int64_t kNumHiddenSize1 = 256, const int64_t kNumHiddenSize2 = 256)
      :
      fc1(kNumInputs, kNumHiddenSize1),
      fc2(kNumHiddenSize1, kNumHiddenSize2),
	   fc3(kNumHiddenSize2, kNumOutputs),
      kNumInputs_(kNumInputs)
	  {
      register_module("fc1", fc1);
      register_module("fc2", fc2);
	  register_module("fc3", fc3);
   }

   torch::Tensor forward(torch::Tensor& x) {
      x = x.view({ -1, kNumInputs_});
      x = torch::relu(fc1->forward(x));
	  x = torch::dropout(x, /*p=*/0.5, /*training=*/is_training());
      x = torch::relu(fc2->forward(x));
	  x = torch::dropout(x, /*p=*/0.5, /*training=*/is_training());
	  x = fc3->forward(x);
      return torch::log_softmax(x, /*dim=*/1);
   }

private:
   torch::nn::Linear fc1;
   torch::nn::Linear fc2;
   torch::nn::Linear fc3;
   int64_t kNumInputs_;

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
