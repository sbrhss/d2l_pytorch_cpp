#include "tools.h"
#include <torch/torch.h>
#include <torch/script.h>


template <typename DataLoader>
float train(
    std::shared_ptr<ResNet>& model,
    size_t epoch,
    DataLoader& data_loader,
    torch::optim::Optimizer& optimizer,
    const size_t dataset_size,
    torch::Device& device) {

   model->train();
   float train_l = 0;
   float train_acc_sum = 0;
   int n = 0;
   auto t1 = std::chrono::high_resolution_clock::now(); 
   for (auto& batch : data_loader) {
      auto data = batch.data.to(device).to(torch::kFloat32), targets = batch.target.to(device);
      optimizer.zero_grad();
      auto output = model->forward(data);
      auto loss = torch::nll_loss(output, targets);
      AT_ASSERT(!std::isnan(loss.template item<float>()));
      loss.backward();
      optimizer.step();
      train_l = loss.template item<float>();
      train_acc_sum += accuracy(output, targets);
      n += batch.data.size(0);
   }
   auto t2 = std::chrono::high_resolution_clock::now();
   auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
   std::cout << "Train Time: " << duration / 1000 << " ms" <<std::endl;
   auto train_acc = train_acc_sum / n * 100;
   std::cout << "epoch: " << epoch<< " ,train loss: " << train_l << " ,train accuracy(%): " << train_acc << std::endl;
   return train_acc;
}

template <typename DataLoader>
float test(
    std::shared_ptr<ResNet>& model,
    size_t epoch,
    DataLoader& data_loader,
    torch::optim::Optimizer& optimizer,
    const size_t dataset_size,
    torch::Device& device) {
   
   model->eval();
   torch::NoGradGuard no_grad;
   float test_l_sum = 0;
   float test_acc_sum = 0;
   int n = 0;
   auto t1 = std::chrono::high_resolution_clock::now();
   for (auto& batch : data_loader) {
      auto data = batch.data.to(device).to(torch::kFloat32), targets = batch.target.to(device);
      auto output = model->forward(data);
      test_l_sum = torch::nll_loss(output, targets).template item<float>();
      test_acc_sum += accuracy(output, targets);
      n += batch.data.size(0);
   }
   auto t2 = std::chrono::high_resolution_clock::now();
   auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
   std::cout << "Test Time: " << duration / 1000 << " ms" <<std::endl;
   auto test_acc = test_acc_sum / n * 100;
   std::cout << "Test loss: " << test_l_sum <<", Test accuracy(%): " << test_acc << std::endl;
   return test_acc;
}