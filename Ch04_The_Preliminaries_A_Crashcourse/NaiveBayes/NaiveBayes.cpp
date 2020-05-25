#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>
#include <memory>

/*

In some form or another, machine learning is all about making predictions.
We might want to predict the probability of a patient suffering a heart attack in the next year, given their clinical history.
In anomaly detection, we might want to assess how likely a set of readings from an airplane's jet engine would be, were it operating normally.
In reinforcement learning, we want an agent to act intelligently in an environment.
This means we need to think about the probability of getting a high reward under each of the available action.
And when we build recommender systems we also need to think about probability. For example, say hypothetically that we work for a large online bookseller.
We might want to estimate the probability that a particular user would buy a particular book. For this we need to use the language of probability and statistics. Entire courses, majors, theses, careers, and even departments, are devoted to probability. So naturally, our goal in this section isn't to teach the whole subject. Instead we hope to get you off the ground, to teach you just enough that you can start building your first machine learning models,
and to give you enough of a flavor for the subject that you can begin to explore it on your own if you wish.

*/

// After how many batches to log a new update with the loss value.
const int64_t kLogInterval = 1000;



torch::Tensor bayespost(const torch::Tensor& data, const torch::Tensor& logpy, const torch::Tensor& logpx, const torch::Tensor& logpxneg);
void plot_mnist(const torch::Tensor& image,
                /*number of rows in a plot, assume a square*/const int num_rows,
                /*number of cols in a plot, assume a square*/const int num_cols);




int main()
{

	using torch::indexing::Slice;
	using torch::indexing::None;

	const uint8_t pixel_rows = 28;
	const uint8_t pixel_cols = 28;
	auto kDataRoot = "../data";


	auto train_dataset = torch::data::datasets::MNIST(kDataRoot);

	//auto images = train_dataset.images();
	//auto slice_of_images = images.index({Slice(0, 100)});
	//plot_mnist(slice_of_images, 10, 10);


	auto mapped_images = train_dataset.map(torch::data::transforms::Normalize<>(0, 1))
	                     .map(torch::data::transforms::Stack<>());


	const size_t train_dataset_size = mapped_images.size().value();

	auto options = torch::TensorOptions().dtype(torch::kFloat32);

	// Initialize the counters
	auto xcount = torch::ones({784, 10}, options);
	auto ycount = torch::ones({10}, options);



	std::cout << "--------------Trianing Phase----------------\n" << std::endl;
	for (size_t i = 0; i < train_dataset_size; ++i) {


		if (i % kLogInterval == 0) {
			std::cout << "Training on " << static_cast<float>(i) / train_dataset_size * 100 << " % is done" << std::endl;
		}

		auto labels = mapped_images.get_batch({i}).target.item<int>();
		ycount.index({labels}) += 1;
		auto sample_image = mapped_images.get_batch({i}).data.reshape({pixel_rows * pixel_cols});
		xcount.index({Slice(), labels}) +=  sample_image;
	}


	std::cout << "--------------Trianing done----------------\n" << std::endl;
	auto logpy 		= torch::log(ycount / torch::sum(ycount, 0));
	auto px 		= xcount / ycount.reshape({1, 10});
	auto logpx 		= torch::log(px);
	auto logpxneg 	= torch::log(1 - px);



	auto test_dataset = torch::data::datasets::MNIST(kDataRoot, torch::data::datasets::MNIST::Mode::kTest);

	//auto images = test_dataset.images();
	//auto slice_of_images = images.index({Slice(0, 100)});
	//plot_mnist(slice_of_images, 10, 10);

	std::cout << "--------------Testing Phase----------------\n" << std::endl;
	auto test_mapped_images = test_dataset.map(torch::data::transforms::Normalize<>(0, 1))
	                          .map(torch::data::transforms::Stack<>());

	const size_t test_dataset_size = test_mapped_images.size().value();
	int err = 0;

	for (size_t i = 0; i < test_dataset_size; ++i) {


		if (i % 1000 == 0) {
			std::cout << "Testing on " << static_cast<float>(i) / test_dataset_size * 100 << " % is done" << std::endl;
		}
		auto labels = test_mapped_images.get_batch({i}).target.item<int>();

		auto sample_image = test_mapped_images.get_batch({i}).data.reshape({pixel_rows * pixel_cols, 1});

		auto post = bayespost(sample_image, logpy, logpx, logpxneg);

		if (post.index({labels}).item<float>() < post.max().item<float>())
			err += 1;

	}

	std::cout << "Naive Bayes has an error rate of: " << static_cast<float>(err) / test_dataset_size * 100 << " %" <<std::endl;


	// Compute the per pixel conditional probabilities

	return 0;
}


void plot_mnist(const torch::Tensor& image,
                /*number of rows in a plot, assume a square*/const int num_rows,
                /*number of cols in a plot, assume a square*/const int num_cols)
{
	using torch::indexing::Slice;
	auto total_num_images = num_rows * num_cols;

	if (image.sizes()[0] == total_num_images) {

		std::cout << "Number of images match :)" << std::endl;
		const int num_pixel = 28; // number of pixels in each rows and columns of MNIST dataset

		torch::Tensor tensors = torch::zeros({num_rows * num_pixel, num_pixel}, torch::TensorOptions().dtype(torch::kFloat32));
		//

		std::vector<cv::Mat> vector_of_Mats;
		vector_of_Mats.reserve(num_rows * num_cols);

		for (int i = 0; i < num_cols; ++i) {
			tensors = image.index({Slice(i * (num_rows), (i + 1)*num_rows)}).reshape({num_rows * num_pixel, num_pixel});
			cv::Mat tensor_to_image(num_rows * num_pixel,  num_pixel, CV_32FC1, tensors.data_ptr());
			vector_of_Mats.push_back(tensor_to_image);
		}

		cv::Mat resultImg(num_rows * num_pixel, num_cols * num_pixel, CV_32FC1, cv::Scalar::all(0));
		cv::hconcat(vector_of_Mats, resultImg);

		cv::namedWindow( "Display window");// Create a window for display.
		cv::imshow( "Display window", resultImg );                   // Show our image inside it.
		cv::waitKey(0);

	}
	else
	{
		std::cerr << "Number of images does not match!" << std::endl;
		exit (EXIT_FAILURE);

	}

}


torch::Tensor bayespost(const torch::Tensor& data, const torch::Tensor& logpy, const torch::Tensor& logpx, const torch::Tensor& logpxneg) {
	// We need to incorporate the prior probability p(y) since p(y|x) is
	// proportional to p(x|y) p(y)

	auto logpost = logpy.clone();

	logpost += torch::sum(logpx * data + logpxneg * (1 - data), 0, false, torch::kFloat32).reshape({10});

	// Normalize to prevent overflow or underflow by subtracting the largest value
	logpost -= torch::max(logpost);
	// Compute the softmax using logpx
	auto post = torch::exp(logpost);
	post /= torch::sum(post);
	return post;
}