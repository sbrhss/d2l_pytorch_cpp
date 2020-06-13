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

torch::nn::Sequential resnet_block(const int kInChannels, 
                                    const int kNumChannels, 
                                    const int kNumResiduals, 
                                    const bool kFirstBlock)
        {
        torch::nn::Sequential net;
        for(int i = 0; i < kNumResiduals; ++i){
            if(i == 0 && !kFirstBlock)
                net->push_back(Residual(kInChannels, kNumChannels, /*kStride=*/2, /*kUse1x1Conv=*/true));
            else
                net->push_back(Residual(kNumChannels, kNumChannels, /*kStride=*/1, /*kUse1x1Conv=*/false)); 
        }
        return net;
}