
#include <torch/torch.h>
#include <torch/script.h>
#include <algorithm>    // std::random_shuffle
#include <vector>       // std::vector
#include "../include/matplotlibcpp.h"
#include "tools.hpp"



/*

The surge of deep learning has inspired the development of a variety of mature software frameworks, 
that automate much of the repetitive work of implementing deep learning models. 
In the previous section we relied only on NDarray for data storage and linear algebra and the auto-differentiation capabilities in the autograd package. 
In practice, because many of the more abstract operations, e.g. data iterators, 
loss functions, model architectures, and optimizers, are so common, deep learning libraries will give us library functions for these as well.

We have used DataLoader to load the MNIST dataset in Section 4.5. In this section, 
we will learn how we can implement the linear regression model in Section 5.2 much more concisely with DataLoader.


*/

namespace plt = matplotlibcpp;
int main () {


	auto option =
	    torch::TensorOptions()
	    .dtype(torch::kFloat32);

	const int num_examples = 1000;
	const int num_inputs = 2;
	const int n_epochs = 50;


	//To start, we will generate the same data set as that used in the previous section.
	auto true_w = torch::tensor({2.0, -3.4}, option);
	auto true_b = 4.2;

	torch::Tensor features = torch::zeros(/*size=*/ {num_examples, num_inputs}, option).normal_(/*mean=*/0, /*std=*/0.01);

	torch::Tensor labels = torch::matmul(features, true_w) + true_b;
	labels += torch::zeros(/*size=*/labels.sizes(), option).normal_(/*mean=*/0, /*std=*/0.01);

	const int batch_size = num_examples / n_epochs;
	auto data_set = customDataset(features, labels).map(torch::data::transforms::Stack<>());

	// In brief, we are loading our data using RandomSampler class which samples randomly
	// For the definition of this function: torch::data::make_data_loader:
	// https://pytorch.org/cppdocs/api/function_namespacetorch_1_1data_1a0d29ca9900cae66957c5cc5052ecc122.html#exhale-function-namespacetorch-1-1data-1a0d29ca9900cae66957c5cc5052ecc122
	auto dataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(data_set), batch_size);


	auto net = std::make_shared<LinearRegressionModel>(2, 1);
	
	// learning rate is 0.01
	// Not surpisingly, we aren’t the first people to implement mini-batch stochastic gradient descent, 
	// and thus torch supports SGD alongside a number of variations on this algorithm through its Trainer class. 
	// When we instantiate the Trainer, we’ll specify the parameters to optimize over (obtainable from our net via net.parameters()), 
	// the optimization algortihm we wish to use (sgd), and a dictionary of hyper-parameters required by our optimization algorithm. 
	// SGD just requires that we set the value learning_rate, (here we set it to 0.03).
	torch::optim::SGD optimizer(net->parameters(), 0.03);

	size_t count = 0;
	std::vector<float> loss_v;
	loss_v.reserve(n_epochs);


	/*
	You might have noticed that expressing our model through torch requires comparatively few lines of code. We didn’t have to individually allocate parameters, 
	define our loss function, or implement stochastic gradient descent. Once we start working with much more complex models, 
	the benefits of relying on torch abstractions will grow considerably. But once we have all the basic pieces in place, 
	the training loop itself strikingly similar to what we did when implementing everything from scratch. To refresh your memory: 
	for some number of epochs, we’ll make a complete pass over the dataset (train_data), grabbing one mini-batch of inputs and corresponding 
	ground-truth labels at a time.

	For each batch, we’ll go through the following ritual:

	• Generate predictions by calling net(X) and calculate the loss l (the forward pass).

	• Calculate gradients by calling l.backward() (the backward pass).

	• Update the model parameters by invoking our SGD optimizer (note that trainer already knows which parameters to optimize over, so we just need to pass in the batch size.

	For good measure, we compute the loss after each epoch and print it to monitor progress.
	*/


	for (auto& batch : *dataloader) {
		auto data = batch.data;
		auto target = batch.target.squeeze();
		// Convert data and target to float32 format
		data = data.to(torch::kF32);
		target = target.to(torch::kF32);
		// Clear the optimizer parameters
		optimizer.zero_grad();

		auto output = net->forward(data);
		// Define the Loss Function
		auto loss = torch::mse_loss(output, target);

		// Backpropagate the loss
		loss.backward();
		// Update the parameters
		optimizer.step();
		++count;
		auto loss_value = loss.mean().item<float>();
		loss_v.emplace_back(loss_value);
		std::cout << "epoch " << count << ", loss " << loss_value << std::endl;
	}

	plt::plot(loss_v);
	plt::title("loss");
	plt::xlim(0, n_epochs);
	plt::grid(true);
	plt::xlabel("epoch");
	plt::ylabel("loss");
	plt::show();
	return 0;

}




