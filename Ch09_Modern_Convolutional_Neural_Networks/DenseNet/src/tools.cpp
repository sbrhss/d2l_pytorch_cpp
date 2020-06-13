#include "tools.h"

float accuracy (const torch::Tensor& y_hat, const torch::Tensor& y) {

	auto compare = (y_hat.argmax(/*dim=*/ 1) == y);
	compare = compare.to(torch::kFloat32);
	return compare.sum().item<float>();
}


//Xavier initialization of weights
void init_weights(torch::nn::Module& module){
if ((typeid(module) == typeid(torch::nn::LinearImpl)) || (typeid(module) == typeid(torch::nn::Linear)) ||
    (typeid(module) == typeid(torch::nn::Conv2dImpl)) || (typeid(module) == typeid(torch::nn::Conv2d))) 
    {
        auto p = module.named_parameters(false);
        auto weight = p.find("weight");
        if (weight != nullptr) {
            torch::nn::init::xavier_normal_(*weight,1);
        }
    }
}

torch::nn::Sequential transition_block(const int kInChannels, 
                                         const int kNumChannels) {

    torch::nn::Sequential tran_net;
    tran_net->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(kInChannels)));
    tran_net->push_back(torch::nn::ReLU());
    tran_net->push_back(torch::nn::Conv2d(
                        torch::nn::Conv2dOptions(kInChannels, kNumChannels, /*kernel_size=*/1)));
    tran_net->push_back(torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(/*kernel_size=*/2).stride(2)));
    //register_module("tran_net", tran_net); 
    return tran_net;
}