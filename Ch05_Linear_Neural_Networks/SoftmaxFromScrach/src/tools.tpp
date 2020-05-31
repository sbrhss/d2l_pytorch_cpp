
#include "tools.h"
#include <torch/torch.h>


template <typename DataLoader>
float train(
   const torch::Tensor& W,
    size_t epoch,
    DataLoader& data_loader,
    size_t kNumInputs,
    const torch::Tensor& b, 
    const float lr,
    const size_t dataset_size ) {


   //model.train();
   float train_l_sum = 0;
   float train_acc_sum = 0;
   int n = 0;
   for (auto& batch : data_loader) {
      auto data = batch.data, targets = batch.target;
      auto output = net(data, W, b, kNumInputs);
      auto loss = cross_entropy(output, targets).sum();
      loss.backward();
      sgd(W, b, lr, batch.data.size(0));
      train_l_sum += loss.template item<float>();
      train_acc_sum += accuracy(output, targets);
      n += batch.data.size(0);

   }
   auto train_acc = train_acc_sum / n * 100;
   std::cout << "epoch: " << epoch << " ,train loss: " <<train_l_sum / n << " ,train accuracy(%): " << train_acc << std::endl;
   return train_acc;
}

template <typename DataLoader>
float test(
   const torch::Tensor& W,
    size_t epoch,
    DataLoader& data_loader,
    size_t kNumInputs,
    const torch::Tensor& b, 
    const size_t dataset_size ) {
   torch::NoGradGuard no_grad;
   float test_l_sum = 0;
   float test_acc_sum = 0;
   int n = 0;
   for (auto& batch : data_loader) {
      auto data = batch.data, targets = batch.target;
      auto output = net(data, W, b, kNumInputs);
      auto loss = cross_entropy(output, targets).sum();
      test_l_sum += loss.template item<float>();
      test_acc_sum += accuracy(output, targets);
      n += batch.data.size(0);

   }
   auto test_acc = test_acc_sum / n * 100;
   std::cout << "Test accuracy(%): " << test_acc << std::endl;
   return test_acc;
}