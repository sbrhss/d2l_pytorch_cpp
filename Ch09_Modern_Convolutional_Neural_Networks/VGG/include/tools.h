

#pragma once

#include <torch/torch.h>
#include <array>

/*
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
		input = input.view({1, 1, 28, 28});
		input = torch::nn::functional::interpolate(input, 
			   torch::nn::functional::InterpolateFuncOptions().size(Resize_).mode(torch::kNearest));
		return input.view({1, 224, 224});

	}

	std::vector<int64_t> Resize_;
};
}
}
}
*/


class Flatten : public torch::nn::Module {
public:
  torch::Tensor forward(const torch::Tensor& input) {
    return input.view({input.sizes()[0], -1});
  }
};

torch::nn::Sequential vgg_block(const int kNumConvs,  int in_channels,  int out_channels);


class VGG : public torch::nn::Module {

public:
	VGG(const std::array<std::array<int, 2>, 4>& conv_arch) 
		  : conv_arch_(conv_arch),
			fc_(
		    /*The fully connected layer part*/
		    Flatten(),
		    torch::nn::Linear(/*in_features=*/512, /*out_features=*/4096),
		    torch::nn::ReLU(),
		    torch::nn::Dropout(/*p=*/0.4),
		    torch::nn::Linear(/*in_features=*/4096, /*out_features=*/4096),
		    torch::nn::ReLU(),
		    torch::nn::Dropout(/*p=*/0.4),
		    torch::nn::Linear(/*in_features=*/4096, /*out_features=*/10)
		)
	{
		const int kModuleSize = conv_arch.size();
		int in_channels = 1;
		for (int i = 0; i < kModuleSize; ++i) {
			int out_channels = conv_arch_[i][1];
			vgg_seq_layers_->extend(*vgg_block(conv_arch_[i][0], in_channels, out_channels));
			in_channels = out_channels;
		}
		
		VGGNetwork_->extend(*vgg_seq_layers_);
		VGGNetwork_->extend(*fc_);
		register_module("VGGNetwork_", VGGNetwork_);

	}

	torch::Tensor forward(torch::Tensor& input) {
		return VGGNetwork_->forward(input);
	}
private:
	std::array<std::array<int, 2>, 4> conv_arch_;
	torch::nn::Sequential vgg_seq_layers_;
	torch::nn::Sequential VGGNetwork_;
	torch::nn::Sequential fc_;
};


float accuracy (const torch::Tensor& y_hat,
                const torch::Tensor& y);

template <typename DataLoader>
float train(
    VGG& model,
    size_t epoch,
    DataLoader& data_loader,
    torch::optim::Optimizer& optimizer,
    const size_t dataset_size );


template <typename DataLoader>
float test(
    VGG& model,
    size_t epoch,
    DataLoader& data_loader,
    torch::optim::Optimizer& optimizer,
    const size_t dataset_size );

