#pragma once

#include <torch/torch.h>
#include <torch/script.h>



/*

When we implemented linear regression from scratch in the previous section, 
we had to define the model parameters and explicitly write out the calculation to produce output using basic linear algebra operations. 
You should know how to do this. But once your models get more complex, even qualitatively simple changes to the model might result in many low-level changes.

We import torch.nn as nn .For standard operations, we can use nn's predefined layers, 
which allow us to focus especially on thelayers used to construct the model rather than having to focus on the implementation. 
To define a linear model, we first import the nn module, which defines a large number of neural network layers 
(note that “nn” is an abbreviation for neural networks). We will first define a model variable net, which is a Sequential instance. 
In nn, a Sequential instance can be regarded as a container that concatenates the various layers in sequence. 
When input data is given, each layer in the container will be calculated in order, and the output of one layer will be the input of the next layer. 
In this example, since our model consists of only one layer, we do not really need Sequential. 
But since nearly all of our future models will involve multiple layers, let’s get into the habit early. 
Recall the architecture of a single layer network. The layer is fully connected since it connects all inputs with all outputs by means of a 
matrix-vector multiplication. In nn, the fully-connected layer is defined in the Linear class. 
Since we only want to generate a single scalar output, we set that number to 1.


*/



class LinearRegressionModel : public torch::nn::Module {
public:
	explicit LinearRegressionModel(int64_t input_size = 2, int64_t output_size = 1) {
		fc = register_module("fc", torch::nn::Linear(input_size, output_size));
	}
	torch::Tensor forward(torch::Tensor x) {
		x = fc->forward(x);
		return x;
	}

	torch::nn::Linear fc{nullptr};
};


class customDataset : public torch::data::Dataset<customDataset>
{
private:
	// Declare 2 tensors for features and labels
	torch::Tensor features_, labels_;

public:
	// Constructor
	explicit customDataset(torch::Tensor features, torch::Tensor labels) {
		features_ = features;
		labels_   = labels;
	};

	torch::data::Example<> get(size_t index) override {
		// You may for example also read in a .csv file that stores locations
		// to your data and then read in the data at this step. Be creative.
		torch::Tensor sample_feature = features_.index({static_cast<int>(index)});
		torch::Tensor sample_label = labels_.index({static_cast<int>(index)});
		return {sample_feature, sample_label};
	};

	torch::optional<size_t> size() const override {
		return labels_.sizes()[0];
	};
};