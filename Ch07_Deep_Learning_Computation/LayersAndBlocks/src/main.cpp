
#include <torch/torch.h>
#include <torch/script.h>


class MLP : public torch::nn::Module {

	// Declare a layer with model parameters. Here, we declare two fully
	// connected layers
public:
	explicit MLP(const int64_t kNumInputs, const int64_t kNumOutputs, const int64_t kNumHiddenSize)

	// Call the constructor of the MLP parent class Module to perform the
	// necessary initialization. In this way, other function parameters can
	// also be specified when constructing an instance, such as the model
	// parameter, params, described in the following sections
		: hidden(torch::nn::Linear(kNumInputs, kNumHiddenSize), torch::nn::ReLU()), // Hidden layer
		  output(kNumHiddenSize, kNumOutputs) // Output layer

	{

		register_module("hidden", hidden);
		register_module("output", output);

	}
	// Define the forward computation of the model, that is, how to return the
	// required model output based on the input x
	torch::Tensor forward(torch::Tensor& x) {
		return output->forward(hidden->forward(x));
	}

private:
	torch::nn::Sequential hidden;
	torch::nn::Linear output;

};


int main () {

	const int64_t kNumHiddenSize = 256;
	const size_t kNumOutputs = 10;
	const size_t kNumInputs = 20;

	std::cout << std::fixed << std::setprecision(4);
	// -------------------------------------- first way of calling sequential
	auto net = torch::nn::Sequential(torch::nn::Linear(kNumInputs, kNumHiddenSize), torch::nn::ReLU(), torch::nn::Linear(kNumHiddenSize, kNumOutputs));
	auto input = torch::randn({2, kNumInputs});
	auto x = net->forward(input);
	std::cout << x << std::endl;

	// -------------------------------------- ineheriting from torch::nn::Module
	MLP Net(kNumInputs, kNumOutputs, kNumHiddenSize);
	std::cout << Net.forward(input) << std::endl;


	// -------------------------------------- using push_back 
	torch::nn::Sequential Network;
	Network->push_back("Linear1", torch::nn::Linear(kNumInputs, kNumHiddenSize));
	Network->push_back("ReLU", torch::nn::ReLU());
	Network->push_back("Linear2", torch::nn::Linear(kNumHiddenSize, kNumOutputs));
	//std::cout << std::endl;
	//std::ostream& os = std::cout;
	//Network->pretty_print(os);


	x = Network->forward(input);
	std::cout << x << std::endl;
	return 0;

}




