
#include <torch/torch.h>
#include <torch/script.h>
#include <vector>       // std::vector
#include "matplotlibcpp.h"
#include "tools.h"
#include "tools.tpp"

namespace plt = matplotlibcpp;
int main () {


	auto kOption =
	    torch::TensorOptions()
	    .dtype(torch::kFloat32);

	// The batch size for training.
	const int64_t kTrainBatchSize = 256;

// The batch size for testing.
	const int64_t kTestBatchSize = 256;

	const size_t kNumInputs = 784;
	const size_t kNumOutputs = 10;

	const char* kDataRoot = "../data";

	std::cout << std::fixed << std::setprecision(4);
	auto train_dataset = torch::data::datasets::MNIST(kDataRoot, torch::data::datasets::MNIST::Mode::kTrain)
	                     .map(torch::data::transforms::Normalize<>(0, 1))
	                     .map(torch::data::transforms::Stack<>());

	const size_t train_dataset_size = train_dataset.size().value();

	auto train_loader =
	    torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
	        std::move(train_dataset), kTrainBatchSize);


	auto test_dataset = torch::data::datasets::MNIST(kDataRoot, torch::data::datasets::MNIST::Mode::kTest)
	                    .map(torch::data::transforms::Normalize<>(0, 1))
	                    .map(torch::data::transforms::Stack<>());

	const size_t test_dataset_size = test_dataset.size().value();

	auto test_loader =
	    torch::data::make_data_loader(std::move(test_dataset), kTestBatchSize);


	auto W = torch::zeros(/*size=*/ {kNumInputs, kNumOutputs}, kOption).normal_(/*mean=*/0, /*std=*/0.01);
	auto b = torch::zeros(/*size=*/kNumOutputs, kOption);

	W.requires_grad_(true);
	b.requires_grad_(true);


	float lr = 0.1;  // Learning rate
	const size_t kNumberOfEpochs = 10;  // Number of iterations
	std::vector<float> train_accuracy_vec, test_accuracy_vec;
	train_accuracy_vec.reserve(kNumberOfEpochs);
	for (size_t num_epochs = 1; num_epochs <= kNumberOfEpochs; ++num_epochs){
		auto train_accuracy = train(W, num_epochs, *train_loader, kNumInputs, b, lr, train_dataset_size);
		train_accuracy_vec.emplace_back(train_accuracy);
		auto test_accuracy = test(W, num_epochs, *test_loader, kNumInputs, b, test_dataset_size);
		test_accuracy_vec.emplace_back(test_accuracy);
	}
	

	plt::plot(train_accuracy_vec, {{"label", "train acc"}});
	plt::plot(test_accuracy_vec, {{"label", "test acc"}});
	plt::title("accuracy (%)");
	plt::xlim(0, static_cast<int>(kNumberOfEpochs));
	plt::grid(true);
	plt::xlabel("epoch");
	plt::ylabel("accuracy");
	plt::legend(); // enable the legend
	plt::show();

	return 0;

}




