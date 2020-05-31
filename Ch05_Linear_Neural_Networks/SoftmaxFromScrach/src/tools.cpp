#include "tools.h"



torch::Tensor softmax(const torch::Tensor& X) {
	auto X_exp = torch::exp(X);
	auto partition = torch::sum(X_exp, /*dim=*/ 1, /*keepdim=*/ true);
	return X_exp / partition;
}


torch::Tensor net(const torch::Tensor& X, const torch::Tensor& W, const torch::Tensor& b, const int kNumInputs) {
	return softmax(torch::matmul(X.reshape({ -1, kNumInputs}), W) + b);
}

torch::Tensor cross_entropy(const torch::Tensor& y_hat, const torch::Tensor& y) {
	return -torch::gather(y_hat, 1, y.unsqueeze(/*dim=*/ 1)).log();
}


float accuracy (const torch::Tensor& y_hat, const torch::Tensor& y) {

	auto compare = (y_hat.argmax(/*dim=*/ 1) == y);
	compare = compare.to(torch::kFloat32);
	return compare.sum().item<float>();
}


void sgd(const torch::Tensor& w, const torch::Tensor& b, const float lr, const size_t batch_size) {

        w.data().sub_(lr * w.grad() / static_cast<float>(batch_size));
        b.data().sub_(lr * b.grad() / static_cast<float>(batch_size));

        w.grad().data().zero_();
        b.grad().data().zero_();

}





