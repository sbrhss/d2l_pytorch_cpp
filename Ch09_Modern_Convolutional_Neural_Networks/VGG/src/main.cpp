#include <torch/torch.h>
#include <torch/script.h>
#include "tools.h"
#include "tools.tpp"


int main () {

  // Where to find the MNIST dataset.
  const char* kDataRoot = "../data";

  // The batch size for training.
  const int64_t kTrainBatchSize = 64;

  // The batch size for testing.
  const int64_t kTestBatchSize = 64;

  // The number of epochs to train.
  const int64_t kNumberOfEpochs = 10;

  torch::manual_seed(1);

    auto cuda_available = torch::cuda::is_available();
    torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
    std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';


  std::array<std::array<int, 2>, 4> conv_arch_shape = { {{2, 64},
                                                          {2, 128},
                                                          {2, 256},
                                                          {2, 512},
                                                        }
  };
  VGG model(conv_arch_shape);
  model.to(device);
  auto train_dataset = torch::data::datasets::MNIST(kDataRoot)
                       .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                       .map(torch::data::transforms::Stack<>());

  
    const size_t train_dataset_size = train_dataset.size().value();
    auto train_loader =
      torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(train_dataset), kTrainBatchSize);

    auto test_dataset = torch::data::datasets::MNIST(
                          kDataRoot, torch::data::datasets::MNIST::Mode::kTest)
                        .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                        .map(torch::data::transforms::Stack<>());
    const size_t test_dataset_size = test_dataset.size().value();
    auto test_loader =
      torch::data::make_data_loader(std::move(test_dataset), kTestBatchSize);

    torch::optim::SGD optimizer(
      model.parameters(), torch::optim::SGDOptions(0.05).momentum(0.9));


    std::vector<float> train_accuracy_vec, test_accuracy_vec;
    train_accuracy_vec.reserve(kNumberOfEpochs);
    for (size_t num_epochs = 1; num_epochs <= kNumberOfEpochs; ++num_epochs) {
      auto train_accuracy = train(model, num_epochs, *train_loader, optimizer, train_dataset_size, device);
      train_accuracy_vec.emplace_back(train_accuracy);
      auto test_accuracy = test(model, num_epochs, *test_loader, optimizer, test_dataset_size, device);
      test_accuracy_vec.emplace_back(test_accuracy);
    }



    return 0;
    
}