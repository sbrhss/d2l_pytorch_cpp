
#include <torch/torch.h>
#include <torch/script.h>

/*

Parameter Management
The ultimate goal of training deep networks is to find good parameter values for a given architecture. 
When everything is standard, the torch.nn.Sequential class is a perfectly good tool for it. 
However, very few models are entirely standard and most scientists want to build things that are novel. 
This section shows how to manipulate parameters. In particular we will cover the following aspects:

Accessing parameters for debugging, diagnostics, to visualize them or to save them is the first step to understanding how to work with custom models.
Secondly, we want to set them in specific ways, e.g. for initialization purposes. We discuss the structure of parameter initializers.
Lastly, we show how this knowledge can be put to good use by building networks that share some parameters.
As always, we start from our trusty Multilayer Perceptron with a hidden layer. This will serve as our choice for demonstrating the various features.

*/



int main () {

	const int64_t kNumHiddenSize = 256;
	const size_t kNumOutputs = 10;
	const size_t kNumInputs = 20;

	std::cout << std::fixed << std::setprecision(4);
	// -------------------------------------- using push_back -------------------
	torch::nn::Sequential Network;
	Network->push_back("Linear1", torch::nn::Linear(kNumInputs, kNumHiddenSize));
	Network->push_back("ReLU", torch::nn::ReLU());
	Network->push_back("Linear2", torch::nn::Linear(kNumHiddenSize, kNumOutputs));



	x = Network->forward(input);
	std::cout << x << std::endl;
	return 0;

}




