

#include "tools.h"



float accuracy (const torch::Tensor& y_hat, const torch::Tensor& y) {

	auto compare = (y_hat.argmax(/*dim=*/ 1) == y);
	compare = compare.to(torch::kFloat32);
	return compare.sum().item<float>();
}

torch::nn::Sequential vgg_block(const int kNumConvs,  int in_channels,  int out_channels) {
	torch::nn::Sequential vgg_layers;
	const int kKernelSize = 3;
	
	for (int i = 0; i < kNumConvs; ++i) {
		vgg_layers->push_back(torch::nn::Conv2d(
					torch::nn::Conv2dOptions(/*in_channels=*/in_channels, /*out_channels=*/out_channels, kKernelSize).padding(/*padding=*/1)));
		vgg_layers->push_back(torch::nn::ReLU());
		in_channels = out_channels;
		
	}
	vgg_layers->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(/*kernel_size=*/{2,2}).stride(2)));
	return vgg_layers;
}



