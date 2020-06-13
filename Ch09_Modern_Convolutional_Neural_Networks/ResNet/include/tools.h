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

void init_weights(torch::nn::Module& module);
torch::nn::Sequential resnet_block(const int kInChannels, 
                                    const int kNumChannels, 
                                    const int kNumResiduals, 
                                    const bool kFirstBlock);

class Residual : public torch::nn::Module {

public:
	   Residual (const int kInChannels, const int kNumChannels, const int kStride, const bool kUse1x1Conv) : 
	
	    kInChannels_(kInChannels), 
	    
	    kNumChannels_(kNumChannels),
	    
	    kStride_(kStride), 
	    
	    kUse1x1Conv_(kUse1x1Conv),
	    
        conv1 (torch::nn::Conv2d(torch::nn::Conv2dOptions
        (kInChannels_, /*out_channels=*/kNumChannels_, /*kernel_size=*/3).stride(kStride_).padding(1))),
        
        conv2 (torch::nn::Conv2d(torch::nn::Conv2dOptions
        (kNumChannels_, /*out_channels=*/kNumChannels_, /*kernel_size=*/3).padding(1))),
        
        conv3 (torch::nn::Conv2d(torch::nn::Conv2dOptions
        (kInChannels_, /*out_channels=*/kNumChannels_, /*kernel_size=*/1).stride(kStride_))),
        
        bn1 (torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(kNumChannels_))),
        
        bn2 (torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(kNumChannels_))),
        
        relu (torch::nn::ReLUOptions().inplace(true))
	{
    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("conv3", conv3);
    register_module("bn1", bn1);
    register_module("bn2", bn2);
    register_module("relu", relu);
	}

	torch::Tensor forward(const torch::Tensor& x){
        auto y = relu->forward(bn1->forward(conv1->forward(x)));
        y = bn2->forward(conv2->forward(y));
        if(kUse1x1Conv_){
            y += conv3->forward(x);
        }else
            y += x; 
       return relu->forward(y);
	}


private:
	const int kInChannels_;
	const int kNumChannels_;
	const int kStride_;
	const bool kUse1x1Conv_;
    torch::nn::Conv2d conv1, conv2, conv3;
	torch::nn::BatchNorm2d bn1, bn2;
	torch::nn::ReLU relu;
};



class ResNet : public torch::nn::Module {
public:
	ResNet() : 
	            b1 (torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 64, /*kernel_size=*/7).stride(2).padding(3)),
                    torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(64)),
                    torch::nn::ReLU(),
                    torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(/*kernel_size=*/3).stride(2).padding(1))),
                    
                    mx2d(torch::nn::AdaptiveMaxPool2dOptions({1, 1})), 
                    
                    linear(torch::nn::LinearOptions(512, 10))
	{
	Network->extend(*b1);    
	Network->extend(*resnet_block(64,64,2,true));
	Network->extend(*resnet_block(64,128,2,false));
    Network->extend(*resnet_block(128,256,2,false));
    Network->extend(*resnet_block(256,512,2,false));
    Network->push_back(mx2d);
    Network->push_back(Flatten());
    Network->push_back(linear);
	register_module("Network", Network);
    register_module("b1", b1);
    register_module("mx2d", mx2d);
    register_module("linear", linear);
	}
	torch::Tensor forward(torch::Tensor& x){
	    x = Network->forward(x);
        return torch::log_softmax(x, /*dim=*/1);
	}

private:
    torch::nn::Sequential Network, b1;
    torch::nn::AdaptiveMaxPool2d mx2d;
    torch::nn::Linear linear;
};



float accuracy (const torch::Tensor& y_hat,
                const torch::Tensor& y);

template <typename DataLoader>
float train(
    std::shared_ptr<ResNet>& model,
    size_t epoch,
    DataLoader& data_loader,
    torch::optim::Optimizer& optimizer,
    const size_t dataset_size,
    torch::Device& device);


template <typename DataLoader>
float test(
    std::shared_ptr<ResNet>& model,
    size_t epoch,
    DataLoader& data_loader,
    torch::optim::Optimizer& optimizer,
    const size_t dataset_size,    
    torch::Device& device);

