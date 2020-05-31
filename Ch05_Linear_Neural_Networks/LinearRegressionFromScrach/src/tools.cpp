#include "tools.hpp"



/*

Reading Data:
Recall that training models, consists of making multiple passes over the dataset,
grabbing one mini-batch of examples at a time and using them to update our model.
Since this process is so fundamental to training machine learning algortihms,
we need a utility for shuffling the data and accessing in mini-batches.


In the following code, we define a data_iter function to demonstrate one possible implementation of this functionality.
The function takes a batch size, a design matrix containing the features, and a vector of labels, yielding minibatches
of size batch_size, each consisting of a tuple of features and labels.


In general, note that we want to use reasonably sized minibatches to take advantage of the GPU hardware,
which excels at parallelizing operations. Because each example can be fed through our models in parallel
and the gradient of the loss function for each example can also be taken in parallel,
GPUs allow us to process hundreds of examples in scarcely more time than it might take to process just a single example.
*/


void data_iter(const int batch_size, const torch::Tensor& features, const torch::Tensor& labels, std::vector<torch::Tensor>& vector_features, std::vector<torch::Tensor>& vactor_labels) {

	const int num_examples = static_cast<int> (features.sizes()[0]);

	std::vector<int> indices = linspace(0, num_examples - 1, num_examples);
	std::random_shuffle ( indices.begin(), indices.end() );

	auto torch_indices = torch::from_blob(indices.data(), {num_examples, 1});

	auto element_num = num_examples / batch_size;
	vector_features.reserve(element_num);
	vactor_labels.reserve(element_num);

	for (int i = 0; i < num_examples; i += batch_size) {
		vector_features.emplace_back(features.index({torch::indexing::Slice(i, i + batch_size - 1)}));
		vactor_labels.emplace_back(labels.index({torch::indexing::Slice(i, i + batch_size - 1)}));
	}
}

// Create a vector of evenly spaced numbers.
template <typename T>
std::vector<T> linspace(T a, T b, size_t N) {
	T h = (b - a) / static_cast<T>(N - 1);
	std::vector<T> xs(N);
	typename std::vector<T>::iterator x;
	T val;
	for (x = xs.begin(), val = a; x != xs.end(); ++x, val += h)
		* x = val;
	return xs;
}


/*

Define the Model
Next, we must define our model, relating its inputs and parameters to its outputs.
Recall that to calculate the output of the linear model, we simply take the matrix-vector dot product of the examples X and the models weights w ,
and add the offset b to each example. Note that below torch::matmul(X, w) is a vector and b is a scalar.
Recall that when we add a vector and a scalar, the scalar is added to each component of the vector.

*/
torch::Tensor linreg(const torch::Tensor& X, const torch::Tensor& w, const torch::Tensor& b) {
	return torch::matmul(X, w) + b;
}


/*

Define the Loss Function
Since updating our model requires taking the gradient of our loss function, we ought to define the loss function first.
Here we will use the squared loss function as described in the previous section.
In the implementation, we need to transform the true value y into the predicted value's shape y_hat.
The result returned by the following function will also be the same as the y_hat shape.

*/


torch::Tensor squared_loss(torch::Tensor& y_hat, torch::Tensor& y) {
	auto diff = y_hat - y.reshape({y_hat.sizes()});
	return diff * diff / 2;
}

/*

Define the Optimization Algorithm
As we discussed in the previous section, linear regression has a closed-form solution.
However, this isn't a book about linear regression, its a book about deep learning. Since none of the other models that this book introduces can be solved analytically,
we will take this opportunity to introduce your first working example of stochastic gradient descent (SGD).

At each step, using one batch randomly drawn from our dataset, we'll estimate the gradient of the loss with respect to our parameters.
Then, we'll update our parameters a small amount in the direction that reduces the loss.

The size of the update step is determined by the learning rate lr.
Because our loss is calculated as a sum over the batch of examples, we normalize our step size by the batch size (batch_size),
so that the magnitude of a typical step size doesn't depend heavily on our choice of the batch size.

*/

void sgd(const torch::Tensor& w, const torch::Tensor& b, const float lr, const size_t batch_size) {

	w.data().sub_(lr * w.grad() / static_cast<float>(batch_size));
	b.data().sub_(lr * b.grad() / static_cast<float>(batch_size));

	w.grad().data().zero_();
	b.grad().data().zero_();

}