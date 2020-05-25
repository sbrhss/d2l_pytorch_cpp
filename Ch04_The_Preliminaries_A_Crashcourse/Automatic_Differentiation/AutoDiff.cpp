#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>

/*

In machine learning, we train models, updating them successively so that they get better and better as they see more and more data.
Usually, getting better means minimizing a loss function, a score that answers the question "how bad is our model?" With neural networks,
we typically choose loss functions that are differentiable with respect to our parameters. Put simply, this means that for each of the model's parameters,
we can determine how much increasing or decreasing it might affect the loss. While the calculations for taking these derivatives are straightforward,
requiring only some basic calculus, for complex models, working out the updates by hand can be a pain (and often error-prone).

*/



// In order to know more about computational graphs, take a look at this video on youtube:
// https://www.youtube.com/watch?v=MswxJw-8PvE&t=612s
int main()
{
	auto options =
	    torch::TensorOptions()
	    .dtype(torch::kFloat32) // float 32
	    .requires_grad(true); // autograd is ON


	auto input = torch::tensor({{0}, {1}, {2}, {3}}, options); // if I use reshape or view, it will give me "undifined output" when I am yaking gradinet

	/*
	Now we are going to compute y and PyTorch will generate a computation graph on the fly.
	Autograd is reverse automatic differentiation system. Conceptually,
	autograd records a graph recording all of the operations that created the data as you execute operations,
	giving you a directed acyclic graph whose leaves are the input tensors and roots are the output tensors.
	By tracing this graph from roots to leaves, you can automatically compute the gradients using the chain rule.

	Note that building the computation graph requires a nontrivial amount of computation.
	So PyTorch will only build the graph when explicitly told to do so. For a tensor to be “recordable”,
	it must be wrapped with torch.autograd.Variable. The Variable class provides almost the same API as Tensor,
	but augments it with the ability to interplay with torch.autograd.Function in order to be differentiated automatically.
	More precisely, a Variable records the history of operations on a Tensor.
	*/

	std::cout << "The tensor is: \n" << input << std::endl;


	auto output = 2 * torch::mm(input.transpose(0, 1), input);

	std::cout << "The tensor is: \n" << output << std::endl;

	output.backward();


	std::cout << "The tensor is: \n" << input.grad() << std::endl;


	//-----------------------------Chain Rule-------------------------------------

	auto x = torch::tensor({{0}, {1}, {2}, {3}}, options);
	y = x * 2;
	auto z = y * x;

	auto head_gradient = torch::tensor({{10.0f}, {1.0f}, {0.1f}, {0.01f}});
	z.backward(head_gradient);
	std::cout << "Gradient is: \n" << x.grad() << std::endl;
	return 0;
}