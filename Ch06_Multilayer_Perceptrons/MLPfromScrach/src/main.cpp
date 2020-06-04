
#include <torch/torch.h>
#include <torch/script.h>
#include <vector>       // std::vector
#include "matplotlibcpp.h"
#include "tools.h"
#include "tools.tpp"



namespace plt = matplotlibcpp;
int main () {


// The batch size for training.
	const int64_t kTrainBatchSize = 256;
// The batch size for testing.
	const int64_t kTestBatchSize = 256;

	const char* kDataRoot = "../data";

	//std::cout << std::fixed << std::setprecision(4);
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


	Net model;

	torch::optim::SGD optimizer(
	    model.parameters(), torch::optim::SGDOptions(0.01).momentum(0.5));

	const size_t kNumberOfEpochs = 20;  // Number of iterations
	std::vector<float> train_accuracy_vec, test_accuracy_vec;
	train_accuracy_vec.reserve(kNumberOfEpochs);
	for (size_t num_epochs = 1; num_epochs <= kNumberOfEpochs; ++num_epochs) {
		auto train_accuracy = train(model, num_epochs, *train_loader, optimizer, kNumInputs, train_dataset_size);
		train_accuracy_vec.emplace_back(train_accuracy);
		auto test_accuracy = test(model, num_epochs, *test_loader, optimizer, kNumInputs, test_dataset_size);
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




