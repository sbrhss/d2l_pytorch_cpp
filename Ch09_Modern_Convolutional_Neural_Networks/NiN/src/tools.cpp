#include "tools.h"

float accuracy (const torch::Tensor& y_hat, const torch::Tensor& y) {

	auto compare = (y_hat.argmax(/*dim=*/ 1) == y);
	compare = compare.to(torch::kFloat32);
	return compare.sum().item<float>();
}

torch::nn::Sequential NiN_block(
    const int kInChannels,
    const int kOutChannels,
    const int kKernelSize,
    const int kStride,
    const int kPadding) {
	torch::nn::Sequential vgg_layers;
	vgg_layers->push_back(torch::nn::Conv2d(
	                          torch::nn::Conv2dOptions(kInChannels, kOutChannels, kKernelSize).stride(kStride).padding(kPadding)));
	vgg_layers->push_back(torch::nn::ReLU());
	vgg_layers->push_back(torch::nn::Conv2d(
	                          torch::nn::Conv2dOptions(kOutChannels, kOutChannels, 1)));
	vgg_layers->push_back(torch::nn::ReLU());
	vgg_layers->push_back(torch::nn::Conv2d(
	                          torch::nn::Conv2dOptions(kOutChannels, kOutChannels, 1)));
	vgg_layers->push_back(torch::nn::ReLU());
	return vgg_layers;
}


//Xavier initialization of weights
void init_weights(torch::nn::Module& module){
if ((typeid(module) == typeid(torch::nn::LinearImpl)) || (typeid(module) == typeid(torch::nn::Linear)) ||
    (typeid(module) == typeid(torch::nn::Conv2dImpl)) || (typeid(module) == typeid(torch::nn::Conv2d))) 
    {
        auto p = module.named_parameters(false);
        auto weight = p.find("weight");
        if (weight != nullptr) {
            torch::nn::init::xavier_normal_(*weight,2);
            //std::cout << *weight << std::endl;
        }
    }
}


