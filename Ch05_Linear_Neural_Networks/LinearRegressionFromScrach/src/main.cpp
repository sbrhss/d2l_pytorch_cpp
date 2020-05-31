
#include <torch/torch.h>
#include <torch/script.h>
#include <algorithm>    // std::random_shuffle
#include <vector>       // std::vector
#include "../include/matplotlibcpp.h"
#include "tools.hpp"



/*

Now that you have some background on the ideas behind linear regression, we are ready to step through a hands-on implementation. 
In this section, and similar ones that follow, we are going to implement all parts of linear regression: 
the data pipeline, the model, the loss function, and the gradient descent optimizer, from scratch. 
Not surprisingly, today's deep learning frameworks can automate nearly all of this work, but if you never learn to implement things from scratch, 
then you may never truly understand how the model works. Moreover, when it comes time to customize models, defining our own layers, loss functions, etc., 
knowing how things work under the hood will come in handy. Thus, we start off describing how to implement linear regression relying only on the primitives 
in the torch.Tensor and autograd packages. In the section immediately following, we will present the compact implementation, 
using all of torch's bells and whistles, but this is where we dive into the details.


*/

namespace plt = matplotlibcpp;
int main () {


	auto option =
	    torch::TensorOptions()
	    .dtype(torch::kFloat32);

	// For this demonstration, we will construct a simple artificial dataset so that we can 
	// easily visualize the data and compare the true pattern to the learned parameters. 
	// We will set the number of examples in our training set to be 1000 and the number of features (or covariates) to 2    
	const size_t num_inputs = 2;
	const int num_examples = 1000;
	// Moreover, to make sure that our algorithm works, 
	// we will assume that the linearity assumption holds with true underlying parameters W=[2,−3.4]T and B = 4.2
	// Thus our synthetic labels will be given according to the following linear model which includes a noise.

	// Following standard assumptions, we choose a noise term 𝜖 
	// that obeys a normal distribution with mean of 0, and in this example, we'll set its standard deviation to  0.01
	auto true_w = torch::tensor({2.0, -3.4}, option);
	auto true_b = 4.2;
	auto features = torch::zeros(/*size=*/ {num_examples, num_inputs}, option).normal_();
	auto labels = torch::matmul(features, true_w) + true_b;
	labels += torch::zeros(/*size=*/labels.sizes(), option).normal_(/*mean=*/0, /*std=*/0.01);


	const int batch_size = 10;

	std::vector<torch::Tensor> vector_features, vactor_labels;

	// Initialize Model Parameters

	// Before we can begin optimizing our model's parameters by gradient descent, we need to have some parameters in the first place. In the following code, 
	// we initialize weights by sampling random numbers from a normal distribution with mean 0 and a standard deviation of 0.01, setting the bias b to 0.

	auto w = torch::zeros(/*size=*/ {num_inputs, 1}, option).normal_(/*mean=*/0, /*std=*/0.01);
	auto b = torch::zeros(/*size=*/ 1, option);

	/*

	Since nobody wants to compute gradients explicitly (this is tedious and error prone), we use automatic differentiation to compute the gradient. 
	See :numref:chapter_autograd for more details. Recall from the autograd chapter that in order for autograd to know that it should store a 
	gradient for our parameters, we need to invoke the attach_grad function, allocating memory to store the gradients that we plan to take.

	*/

	w.requires_grad_(true);
	b.requires_grad_(true);


	float lr = 0.5;  // Learning rate
	int num_epochs = 100;  // Number of iterations

	data_iter(batch_size, features, labels, vector_features, vactor_labels);


	/*

	Training:
	Now that we have all of the parts in place, we are ready to implement the main training loop. 
	It is crucial that you understand this code because you will see training loops that are nearly identical to this one over and over again 
	throughout your career in deep learning.

	In each iteration, we will grab minibatches of models, first passing them through our model to obtain a set of predictions. 
	After calculating the loss, we will call the backward function to backpropagate through the network, storing the gradients with respect to 
	each parameter in its corresponding .grad attribute. Finally, we will call the optimization algorithm sgd to update the model parameters. 
	Since we previously set the batch size batch_size to 10, the loss shape l for each small batch is (10, 1).

	*/

	std::vector<float> loss_vec;
	loss_vec.reserve(num_epochs);
	for (int i = 0; i < num_epochs; ++i) {
		auto net = linreg(vector_features[i], w, b);
		auto loss = squared_loss(net, vactor_labels[i]);
		loss.mean().backward();
		sgd(w, b, lr, batch_size);
		torch::NoGradGuard guard;

		auto net_input = linreg(features, w, b);
		auto train_l = squared_loss(net_input, labels);

		auto loss_val = train_l.mean().item<float>();
		loss_vec.emplace_back(loss_val);
		std::cout << "epoch " << i + 1 << " " << ",loss "
		          << loss_val << std::endl;

	}

	plt::plot(loss_vec);
	plt::title("loss");
	plt::xlim(0, num_epochs);
	plt::grid(true);
	plt::xlabel("epoch");
	plt::ylabel("loss");
	plt::show();

	return 0;

}




