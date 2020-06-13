#pragma once

#include <torch/torch.h>
#include <array>
#include <string>
#include <iostream>

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
torch::nn::Sequential transition_block(const int kInChannels, 
                                        const int kNumChannels);
void init_weights(torch::nn::Module& module);

class conv_block: public torch::nn::Module{ 
public :
    conv_block(const int kInChannels, const int kNumChannels) :
    conv_net(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(kInChannels)),
        torch::nn::ReLU(),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(kInChannels, kNumChannels, /*kernel_size=*/3).padding(1))) {register_module("conv_net", conv_net);}
        
        torch::Tensor forward(const torch::Tensor& x){
            return conv_net->forward(x);
        }
private:
        torch::nn::Sequential conv_net;
};



class Flatten : public torch::nn::Module {
public:
	torch::Tensor forward(const torch::Tensor& input) {
		return input.view({input.sizes()[0], -1});
	}
};


class Reshape : public torch::nn::Module {
public:
    torch::Tensor forward(const torch::Tensor& input) {
        return input.view({-1, 1, 96, 96});
    }
};


class DenseBlock : public torch::nn::Module {

public:
	DenseBlock (const int kNumConvs, const int kInChannels, const int kNumChannels) : kNumConvs_(kNumConvs), kInChannels_(kInChannels), kNumChannels_(kNumChannels)
	{
        for(int i = 0; i < kNumConvs_; ++i){
           dense_block_->push_back(conv_block(kNumChannels_ * i + kInChannels_, kNumChannels_));
        }
        register_module("dense_block_", dense_block_);
	}

	torch::Tensor forward(const torch::Tensor& x){
        //Concatenate the input and output of each block on the channel dimension
       torch::Tensor out = x;
       for(auto& net : *dense_block_) {
            auto y = net.forward(out);
            out = torch::cat({out, y}, /*dim=*/1);
       }
       return out;
	}


private:
    int kNumConvs_;
	int kInChannels_;
	int kNumChannels_;
    torch::nn::Sequential dense_block_;
};

class DenseNet : public torch::nn::Module {
public:
	DenseNet()
    {   
        Network->push_back(Reshape());
        Network->push_back(torch::nn::Conv2d(
                torch::nn::Conv2dOptions(/*in_channels=*/1, /*out_channels=*/64, /*kernel_size=*/7).stride(2).padding(3)));
        Network->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(64)));
        Network->push_back(torch::nn::ReLU());
        Network->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(/*kernel_size=*/3).stride(2).padding(1)));
        // Num_channels: the current number of channels
        int num_channels = 64;
        const int growth_rate = 32;
        std::array<int, 4> num_convs_in_dense_blocks = {4, 4, 4, 4};
        for (unsigned int i = 0; i < num_convs_in_dense_blocks.size(); ++i){
            //------------------------------------------------------------------------------------//
           Network->push_back(DenseBlock(num_convs_in_dense_blocks[i], num_channels, growth_rate));
           //-------------------------------------------------------------------------------------//
            // This is the number of output channels in the previous dense block
            num_channels += num_convs_in_dense_blocks[i] * growth_rate;
            // A transition layer that haves the number of channels is added between the dense blocks
            if(i != num_convs_in_dense_blocks.size()-1){
                Network->extend(*transition_block(num_channels, num_channels/2));
                num_channels /= 2;
            }
        }
        Network->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(num_channels)));
        Network->push_back(torch::nn::ReLU());
        Network->push_back(torch::nn::AdaptiveMaxPool2d(torch::nn::AdaptiveMaxPool2dOptions({1,1})));
        Network->push_back(Flatten());
        Network->push_back(torch::nn::Linear(num_channels, 10));

        register_module("NetworkX", Network);
	}

	torch::Tensor forward(torch::Tensor& x){
	    x = Network->forward(x);
        return torch::log_softmax(x, /*dim=*/1);
	}

private:
    torch::nn::Sequential Network;

};



float accuracy (const torch::Tensor& y_hat,
                const torch::Tensor& y);

template <typename DataLoader>
float train(
    std::shared_ptr<DenseNet>& model,
    size_t epoch,
    DataLoader& data_loader,
    torch::optim::Optimizer& optimizer,
    const size_t dataset_size,
    torch::Device& device);


template <typename DataLoader>
float test(
    std::shared_ptr<DenseNet>& model,
    size_t epoch,
    DataLoader& data_loader,
    torch::optim::Optimizer& optimizer,
    const size_t dataset_size,    
    torch::Device& device);

