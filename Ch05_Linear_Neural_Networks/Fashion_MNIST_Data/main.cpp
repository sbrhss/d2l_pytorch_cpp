#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>
#include <memory>

/*

Before we implement softmax regression ourselves, let's pick a real dataset to work with. 
To make things visually compelling, we will pick an image classification dataset. 
The most commonly used image classification data set is the MNIST handwritten digit recognition data set, proposed by LeCun, Cortes and Burges in the 1990s. 
However, even simple models achieve classification accuracy over 95% on MNIST, so it is hard to spot the differences between better models and weaker ones. 
In order to get a better intuition, we will use the qualitatively similar, but comparatively complex Fashion-MNIST dataset, proposed by Xiao, Rasul and Vollgraf in 2017.

*/

// After how many batches to log a new update with the loss value.
const int64_t kLogInterval = 1000;

void plot_mnist(const torch::Tensor& image,
                /*number of rows in a plot, assume a square*/const int num_rows,
                /*number of cols in a plot, assume a square*/const int num_cols);




int main()
{

	using torch::indexing::Slice;
	using torch::indexing::None;

	//const uint8_t pixel_rows = 28;
	//const uint8_t pixel_cols = 28;
	auto kDataRoot = "../data";


	auto train_dataset = torch::data::datasets::MNIST(kDataRoot);

	auto train_images = train_dataset.images();
	auto slice_of_images_train = train_images.index({Slice(0, 100)});
	plot_mnist(slice_of_images_train, 10, 10);


	auto options = torch::TensorOptions().dtype(torch::kFloat32);

	auto xcount = torch::ones({784, 10}, options);
	auto ycount = torch::ones({10}, options);

	auto test_dataset = torch::data::datasets::MNIST(kDataRoot, torch::data::datasets::MNIST::Mode::kTest);

	auto test_images = test_dataset.images();
	auto slice_of_images_test = test_images.index({Slice(0, 100)});
	plot_mnist(slice_of_images_test, 10, 10);




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