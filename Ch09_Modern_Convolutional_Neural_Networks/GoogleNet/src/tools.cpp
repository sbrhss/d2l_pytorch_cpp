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


