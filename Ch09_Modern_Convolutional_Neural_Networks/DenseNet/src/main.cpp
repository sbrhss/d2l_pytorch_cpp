#include <torch/torch.h>
#include <torch/script.h>
#include "tools.h"
#include "tools.tpp"
#include <chrono>

int main () {

  // Where to find the MNIST dataset.
  const char* kDataRoot = "../data";

  // The batch size for training.
  const int64_t kTrainBatchSize = 64;

  // The batch size for testing.
  const int64_t kTestBatchSize = 1024;

  // The number of epochs to train.
  const int64_t kNumberOfEpochs = 2;

  //torch::manual_seed(1);
  auto cuda_available = torch::cuda::is_available();
  torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
  std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

  auto model = std::make_shared<DenseNet>();
  model->apply(init_weights);
  model->to(device);

    auto train_dataset = torch::data::datasets::MNIST(kDataRoot)
                         .map(torch::data::transforms::Resize<>({96, 96}))
                         .map(torch::data::transforms::Normalize<>(0, 1))
                         .map(torch::data::transforms::Stack<>());

    const size_t train_dataset_size = train_dataset.size().value();

    auto train_loader =
      torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(train_dataset), torch::data::DataLoaderOptions().batch_size(kTrainBatchSize).workers(4));

    auto test_dataset = torch::data::datasets::MNIST(
                          kDataRoot, torch::data::datasets::MNIST::Mode::kTest)
                        .map(torch::data::transforms::Resize<>({96, 96}))
                        .map(torch::data::transforms::Normalize<>(0, 1))
                        .map(torch::data::transforms::Stack<>());

    const size_t test_dataset_size = test_dataset.size().value();

    auto test_loader = torch::data::make_data_loader(std::move(test_dataset), torch::data::DataLoaderOptions().batch_size(kTestBatchSize).workers(4));



    torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(0.01).momentum(0.5));

    std::vector<float> train_accuracy_vec, test_accuracy_vec;
    train_accuracy_vec.reserve(kNumberOfEpochs);
    for (size_t num_epochs = 1; num_epochs <= kNumberOfEpochs; ++num_epochs) {
      auto train_accuracy = train(model, num_epochs, *train_loader, optimizer, train_dataset_size, device);
      auto test_accuracy = test(model, num_epochs, *test_loader, optimizer, test_dataset_size, device);
      test_accuracy_vec.emplace_back(test_accuracy);
      train_accuracy_vec.emplace_back(train_accuracy);
    }

  return 0;

}
