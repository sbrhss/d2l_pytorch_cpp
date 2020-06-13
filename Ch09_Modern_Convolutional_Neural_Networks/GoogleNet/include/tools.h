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

class Inception : public torch::nn::Module {

public:
    //Inception(){}
	Inception(const int kInChannels, const int c1, const std::array<int, 2>& c2, const std::array<int, 2>& c3, const int c4) : 
	    // Path 1 is a single 1 x 1 convolutional layer
        p1_1 (torch::nn::Conv2d(torch::nn::Conv2dOptions(kInChannels, /*out_channels=*/c1, /*kernel_size=*/1))),
        // Path 2 is a 1 x 1 convolutional layer followed by a 3 x 3
        // convolutional layer
        p2_1 (torch::nn::Conv2d(torch::nn::Conv2dOptions(kInChannels, /*out_channels=*/c2[0], /*kernel_size=*/1))),
        p2_2 (torch::nn::Conv2d(torch::nn::Conv2dOptions(/*in_channels=*/c2[0], /*out_channels=*/c2[1], /*kernel_size=*/3).padding(1))),
        // Path 3 is a 1 x 1 convolutional layer followed by a 5 x 5
        // convolutional layer
        p3_1 (torch::nn::Conv2d(torch::nn::Conv2dOptions(/*in_channels=*/kInChannels, /*out_channels=*/c3[0], /*kernel_size=*/1))),
        p3_2 (torch::nn::Conv2d(torch::nn::Conv2dOptions(/*in_channels=*/c3[0], /*out_channels=*/c3[1], /*kernel_size=*/5).padding(2))),
        // Path 4 is a 3 x 3 maximum pooling layer followed by a 1 x 1
        // convolutional layer
        p4_2 (torch::nn::Conv2d(torch::nn::Conv2dOptions(kInChannels, /*out_channels=*/c4, /*kernel_size=*/1))),
        p4_1 (torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(/*kernel_size=*/3).stride(1).padding(1)))
	{
    register_module("p1_1", p1_1);
    register_module("p2_1", p2_1);
    register_module("p2_2", p2_2);
    register_module("p3_1", p3_1);
    register_module("p3_2", p3_2);
    register_module("p4_1", p4_1);
    register_module("p4_2", p4_2);
	}

	torch::Tensor forward(const torch::Tensor& x)  {
        auto p1 = torch::nn::functional::relu(p1_1->forward(x));
        auto p2 = torch::nn::functional::relu(p2_2->forward(torch::nn::functional::relu(p2_1->forward(x))));
        auto p3 = torch::nn::functional::relu(p3_2->forward(torch::nn::functional::relu(p3_1->forward(x))));
        auto p4 = torch::nn::functional::relu(p4_2->forward(p4_1->forward(x)));
         //Concatenate the outputs on the channel dimension
       return torch::cat({p1, p2, p3, p4}, /*dim=*/1);
	}


private:
    torch::nn::Conv2d p1_1, p2_1, p2_2, p3_1, p3_2, p4_2;
	torch::nn::MaxPool2d p4_1;
};

class GoogleNet : public torch::nn::Module {
public:
	GoogleNet() : 

    b1 (torch::nn::Conv2d(torch::nn::Conv2dOptions(/*in_channels=*/1, /*out_channels=*/64, /*kernel_size=*/7).stride(2).padding(3)),
        torch::nn::ReLU(),
        torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(/*kernel_size=*/3).stride(2).padding(1))),
                   
    b2 (torch::nn::Conv2d(torch::nn::Conv2dOptions(/*in_channels=*/64, /*out_channels=*/64, /*kernel_size=*/1)),
        torch::nn::ReLU(),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(/*in_channels=*/64, /*out_channels=*/192, /*kernel_size=*/3).padding(1)),
        torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(/*kernel_size=*/3).stride(2).padding(1))),
     
    b3 (Inception(192, 64, {96, 128}, {16, 32}, 32), 
        Inception(256, 128, {128, 192}, {32, 96}, 64),
        torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(/*kernel_size=*/3).stride(2).padding(1))),
         
    b4 (Inception(480, 192, {96, 208}, {16, 48}, 64),
        Inception(512, 160, {112, 224}, {24, 64}, 64),
        Inception(512, 128, {128, 256}, {24, 64}, 64),
        Inception(512, 112, {144, 288}, {32, 64}, 64),
        Inception(528, 256, {160, 320}, {32, 128}, 128),
        torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(/*kernel_size=*/3).stride(2).padding(1))),
        
    b5 (Inception(832, 256, {160, 320}, {32, 128}, 128),
        Inception(832, 384, {192, 384}, {48, 128}, 128),
        torch::nn::AdaptiveMaxPool2d(torch::nn::AdaptiveMaxPool2dOptions({1,1})),
        Flatten(),
        torch::nn::Linear(1024, 10))
    
	{
	register_module("b1", b1);
	register_module("b2", b2);
	register_module("b3", b3);
	register_module("b4", b4);
	register_module("b5", b5);
	}

	torch::Tensor forward(torch::Tensor& x){
	    x = b1->forward(x);
	    x = b2->forward(x);
	    x = b3->forward(x);
	    x = b4->forward(x);
	    x = b5->forward(x);
        return torch::log_softmax(x, /*dim=*/1);
	}

private:
    torch::nn::Sequential b1, b2, b3, b4, b5;
};



float accuracy (const torch::Tensor& y_hat,
                const torch::Tensor& y);

template <typename DataLoader>
float train(
    std::shared_ptr<GoogleNet>& model,
    size_t epoch,
    DataLoader& data_loader,
    torch::optim::Optimizer& optimizer,
    const size_t dataset_size,
    torch::Device device);


template <typename DataLoader>
float test(
    std::shared_ptr<GoogleNet>& model,
    size_t epoch,
    DataLoader& data_loader,
    torch::optim::Optimizer& optimizer,
    const size_t dataset_size,    
    torch::Device device);

