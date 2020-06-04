
#include "tools.h"
#include <torch/torch.h>
#include <torch/script.h>


template <typename DataLoader>
float train(
    Net& model,
    size_t epoch,
    DataLoader& data_loader,
    torch::optim::Optimizer& optimizer,
    size_t kNumInputs,
    const size_t dataset_size ) {

   model.train();
   float train_l = 0;
   float train_acc_sum = 0;
   int n = 0;
    
   for (auto& batch : data_loader) {

      auto data = batch.data, targets = batch.target;
      optimizer.zero_grad();
      auto output = model.forward(data);
      auto loss = torch::nll_loss(output, targets);
      AT_ASSERT(!std::isnan(loss.template item<float>()));
      loss.backward();
      optimizer.step();
      train_l = loss.template item<float>();
      train_acc_sum += accuracy(output, targets);
      n += batch.data.size(0);
   }
   
   auto train_acc = train_acc_sum / n * 100;
   std::cout << "epoch: " << epoch << " ,train loss: " << train_l << " ,train accuracy(%): " << train_acc << std::endl;
   return train_acc;
}

template <typename DataLoader>
float test(
    Net& model,
    size_t epoch,
    DataLoader& data_loader,
    torch::optim::Optimizer& optimizer,
    size_t kNumInputs,
    const size_t dataset_size ) {
   torch::NoGradGuard no_grad;
   model.eval();
   float test_l_sum = 0;
   float test_acc_sum = 0;
   int n = 0;
   for (auto& batch : data_loader) {
      auto data = batch.data, targets = batch.target;
      auto output = model.forward(data);
      test_l_sum = torch::nll_loss(output, targets).template item<float>();
      test_acc_sum += accuracy(output, targets);
      n += batch.data.size(0);

   }
   auto test_acc = test_acc_sum / n * 100;
   std::cout << "Test loss: " << test_l_sum <<", Test accuracy(%): " << test_acc << std::endl;
   return test_acc;
}