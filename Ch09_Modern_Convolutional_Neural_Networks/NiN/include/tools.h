

#pragma once

#include <torch/torch.h>
#include <array>


namespace torch {
namespace data {
namespace transforms {
/// Resizes input tensors by the given sizes
template <typename Target = Tensor>
struct Resize : public TensorTransform<Target> {
	/// Constructs a `Resize` transform.
	Resize(std::vector<int64_t> Resize)
		: Resize_(Resize) {}

	torch::Tensor operator()(Tensor input) {
		input = input.unsqueeze(0);
		input = torch::nn::functional::interpolate(input,
			   torch::nn::functional::InterpolateFuncOptions().size(Resize_).mode(torch::kNearest));
		input = input.squeeze(0);	   
		return input;

	}

	std::vector<int64_t> Resize_;
};
}
}
}



class Flatten : public torch::nn::Module {
public:
	torch::Tensor forward(const torch::Tensor& input) {
		return input.view({input.sizes()[0], -1});
	}
};

torch::nn::Sequential NiN_block(
    const int kInChannels,
    const int kOutChannels,
    const int kKernelSize,
    const int kStride,
    const int kPadding);

void init_weights(torch::nn::Module& module);

class NiN : public torch::nn::Module {

public:
	NiN() :
		n1 ( NiN_block(1,/*out_channels=*/96, /*kernel_size=*/11, /*strides=*/4, /*padding=*/0)),
		n2 ( NiN_block(96,/*out_channels=*/256, /*kernel_size=*/5, /*strides=*/1, /*padding=*/2)),
		n3 ( NiN_block(256,/*out_channels=*/384, /*kernel_size=*/3, /*strides=*/1, /*padding=*/1)),
		n4 ( NiN_block(384,/*out_channels=*/10, /*kernel_size=*/3, /*strides=*/1, /*padding=*/1) ),
        m1 ( torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2))),    
        m2 ( torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2))),
        m3 ( torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2))),
        dropout1 ( torch::nn::Dropout2d(0.5) ),
        //Global Average Pooling can be achieved by AdaptiveMaxPool2d with output size = (1,1)
        avg1 ( torch::nn::AdaptiveMaxPool2d(torch::nn::AdaptiveMaxPool2dOptions({1, 1})) )
       
	{
	
    register_module("n1", n1);
    register_module("n2", n2);
    register_module("n3", n3);
    register_module("n4", n4);
    register_module("m1", m1);
    register_module("m2", m2);
    register_module("m3", m3);
    register_module("dropout1", dropout1);
    register_module("avg1", avg1);
	}

	torch::Tensor forward(torch::Tensor& input) {
		input = n1->forward(input);
		input = m1->forward(input);
		input = n2->forward(input);
		input = m2->forward(input);
		input = n3->forward(input);
		input = m3->forward(input);
		input = dropout1->forward(input);
		input = n4->forward(input);
		input = avg1->forward(input);
		return Flatten().forward(input);
	}

private:
	torch::nn::Sequential n1, n2, n3, n4;
	torch::nn::MaxPool2d m1, m2, m3;
	torch::nn::Dropout2d dropout1;
	torch::nn::AdaptiveMaxPool2d avg1;
};



float accuracy (const torch::Tensor& y_hat,
                const torch::Tensor& y);

template <typename DataLoader>
float train(
    NiN& model,
    size_t epoch,
    DataLoader& data_loader,
    torch::optim::Optimizer& optimizer,
    const size_t dataset_size,
    torch::Device device);


template <typename DataLoader>
float test(
    NiN& model,
    size_t epoch,
    DataLoader& data_loader,
    torch::optim::Optimizer& optimizer,
    const size_t dataset_size,    
    torch::Device device);

