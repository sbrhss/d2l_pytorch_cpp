#include <torch/torch.h>
#include <torch/script.h>
#include <array>
#include <vector>
#include <iostream>
#include <iomanip>

/*

Data Manipulation:
It is impossible to get anything done if we cannot manipulate data.
Generally, there are two important things we need to do with data: (i) acquire it and (ii) process it once it is inside the computer.
There is no point in acquiring data if we do not even know how to store it, so let's get our hands dirty first by playing with synthetic data.
We will start by introducing the tensor, PyTorch's primary tool for storing and transforming data.
Tensors support asynchronous computation on CPU, GPU and provide support for automatic differentiation.

*/

torch::Dimname dimnameFromString(const std::string& str);

int main()
{

	std::cout << "-----------------------------------------------------------------------"  << std::endl;
	std::cout << "---------------------------Data Manipulation---------------------------"  << std::endl;
	std::cout << "----------------Getting Started: "  										<< std::endl;

	std::cout << std::fixed << std::setprecision(4);


	// Tensors represent (possibly multi-dimensional) arrays of numerical values. The simplest object we can create is a vector.
	// To start, we can use arange to create a row vector with 12 consecutive integers.

	auto x = torch::arange(/*range of 12 numbers*/12, /*float 32 bits*/torch::kFloat32);
	std::cout << "Tensor is here: \n" << x << std::endl;

	auto size_tensor = x.sizes();
	std::cout << "The shape of tensor is: \n" << size_tensor << std::endl;

	/*We use the reshape function to change the shape of one (possibly multi-dimensional) array,
	to another that contains the same number of elements. For example, we can transform the shape of our line vector x to (3, 4),
	which contains the same values but interprets them as a matrix containing 3 rows and 4 columns.
	Note that although the shape has changed, the elements in x have not.*/
	auto x_reshaped = x.reshape({/*rows*/3, /*columns*/4});
	std::cout << "Reshaped Tensor is here: \n" << x_reshaped << std::endl;

	/*
	Reshaping by manually specifying each of the dimensions can get annoying.
	Once we know one of the dimensions, why should we have to perform the division our selves to determine the other?
	For example, above, to get a matrix with 3 rows, we had to specify that it should have 4 columns
	(to account for the 12 elements). Fortunately, PyTorch can automatically work out one dimension given the other.
	We can invoke this capability by placing -1 for the dimension that we would like PyTorch to automatically infer. In our case, instead of x::reshape((3, 4)),
	we could have equivalently used x::reshape((-1, 4)) or x::reshape((3, -1)).
	*/



	x_reshaped = x.reshape({/*2 rows*/2, /*calculate the columns*/ -1});
	std::cout << "Reshaped Tensor is here: \n" << x_reshaped << std::endl;


	//  As the name suggests torch.empty() makes an empty tensor;
	std::cout << "Define a float Tensor: \n" << torch::empty({/*rows*/2, /*columns*/3}) << std::endl;


	// setting options of a tensor.
	auto options =
	    torch::TensorOptions()
	    .dtype(torch::kFloat32);


	//  Perhaps most often, we want an array of all zeros. To create tensor with all elements set to 0 and a shape of (2, 3, 4) we can invoke:
	auto zeros_tensor = torch::zeros({/*row*/2, /*column*/3,/*depth*/ 4}, options);
	// torch::Tensor zero_tensor = torch::zeros({/*row*/2, /*column*/3,/*depth*/ 4},   	torch::TensorOptions().dtype(torch::kFloat32));
	std::cout << "Zeros Tensor: \n" << zeros_tensor << std::endl;


	// We can create tensors with each element set to 1 works via
	auto ones_tensor = torch::ones({/*row*/2, /*column*/3,/*depth*/ 4}, options);
	// torch::Tensor zero_tensor = torch::zeros({/*row*/2, /*column*/3,/*depth*/ 4},   	torch::TensorOptions().dtype(torch::kFloat32));
	std::cout << "Ones Tensor: \n" << 2 * ones_tensor << std::endl;

	// We can also specify the value of each element in the desired NDArray by supplying a list containing the numerical values.
	auto arbitrary_tensor = torch::tensor({/*1st row*/{1, 2, 3},/*2nd row*/{4, 5, 6}}, options);
	std::cout << "Arbitrary Tensor: \n" << arbitrary_tensor << std::endl;

	std::cout << "-----------------------------------------------------------------------"  << std::endl;
	std::cout << "-----------------------------------------------------------------------"  << std::endl;
	std::cout << "----------------Operations: "		  										<< std::endl;

	x = arbitrary_tensor;
	auto y = torch::ones_like(x, options);

	std::cout << "Make a variable like x, full of one: \n" << y << std::endl;

	std::cout << "------------------------Adding two Tensors-------------------------"  << std::endl;
	// adding 2 tensors, the first way
	std::cout << "x + y: \n" << x + y << std::endl;
	// adding 2 tensors, the second way
	std::cout << "x + y= \n" << x.add(y) << std::endl;
	// print x after operation
	std::cout << "x= \n" << x << std::endl;
	// adding 2 tensors, the third way
	std::cout << "x + y= \n" << add(x, y) << std::endl;

	std::cout << "------------------------subtracting two Tensors-------------------------"  << std::endl;
	// the same also applys for subtracting
	// subtracting 2 tensors, the first way
	std::cout << "x - y: \n" << x - y << std::endl;
	// subtracting 2 tensors, the second way
	std::cout << "x - y= \n" << x.sub(y) << std::endl;
	// subtracting 2 tensors, the third way
	std::cout << "x - y= \n" << sub(x, y) << std::endl;

	std::cout << "------------------------Elementwise multiplication of two tensors:"  << std::endl;
	// element wise multiplication of 2 tensors:
	std::cout << "x * y: \n" << x * y << std::endl;

	std::cout << "------------------------Elementwise division of two tensors:"  << std::endl;
	// element wise multiplication of 2 tensors:
	std::cout << "x / y: \n" << x / y << std::endl;

	std::cout << "------------------------Elementwise exponension:"  << std::endl;
	// static Tensor at::exp(const Tensor &self)
	auto x_exp = x.exp();
	std::cout << "exp(1, 2, 3)= \n" << x_exp << std::endl;




	x = torch::arange(12, options).reshape({/*rows*/3, /*columns*/4});
	y = torch::tensor({{2, 1, 4, 3},
						{1, 2, 3, 4},
						{4, 3, 2, 1}
					},
					torch::TensorOptions().dtype(torch::kFloat32));

	std::cout << "------------------------Getting data type:"  << std::endl;
	std::cout << "x.dtype= " << x.dtype() << std::endl;

	std::cout << "------------------------Matrix Multiplication ------------------------- \n"  << std::endl;
	// matrix multiplication
	auto muliplication = x.mm(y.transpose(0, 1));
	std::cout << "X * Y: \n" << muliplication << std::endl;



	std::cout << "------------------------Named Tensors:"  << std::endl;
	std::cout << "This part, for now, is not completely developed by Pytorch group!"  << std::endl;
	std::cout << "-----------------------------------------------------------------------"  << std::endl;
	std::cout << "-----------------------------------------------------------------------"  << std::endl;


	x = torch::arange(9, options).reshape({/*rows*/3, /*columns*/3});

	std::cout << x.names() << std::endl;
	// concatinate arrays
	std::array<torch::Tensor, 2> arrayRef = {x , x};

	auto cat_out = at::cat(arrayRef, 0);
	std::cout << "Concatinate x with x: \n" << cat_out << std::endl;


	std::cout << "-------------------------------- \n";
	auto r = dimnameFromString("r");
	auto c = dimnameFromString("c");
	std::vector<torch::Dimname> names = { r, c };

	auto z_ = torch::zeros({5, 5}, names, options);
	std::array<torch::Tensor, 2> arrayRefz_ = {z_, z_};

	auto cat_out_z = at::cat(arrayRefz_, c);
	std::cout << z_.names() << std::endl;

	std::cout << "Concatinate z_ with z_: \n" << cat_out_z  << std::endl;

	std::cout << "-----------------------------------------------------------------------"  << std::endl;
	std::cout << "-----------------------------------------------------------------------"  << std::endl;


	std::cout << "------------------------Comulative Sum:"  << std::endl;
	options =
	    torch::TensorOptions()
	    .dtype(torch::kUInt8);

	x = torch::ones(10, options);
	auto x_sum = x.cumsum(0, torch::kUInt8);

	std::cout << x_sum << std::endl;
	std::cout << x_sum.index({9}).item<float>() << std::endl;



	//Broadcast Mechanism
	auto a = torch::arange(3, options);

	std::cout << "Sum of le= \n" << a.sum() << std::endl;

	std::cout << "------------------------------ Indexing and Slicing ------------------------------:"  << std::endl;
	using torch::indexing::Slice;
	using torch::indexing::None;
	using torch::indexing::Ellipsis;

	x = torch::tensor({{2, 1, 4, 3},
						{1, 2, 3, 4},
						{4, 3, 2, 1},
						{8, 7, 3, 1}
					}, torch::TensorOptions().dtype(torch::kFloat32));
	// x =
	// 2 1 4 3
	// 1 2 3 4
	// 4 2 2 1

	// Extract a single element tensor:
	std::cout << "\"x[0,2]\" as tensor:\n" << x.index({0, 2}) << '\n';
	// Output:
	// tensor [4.0]
	std::cout << "\"x[0,2]\" as value:\n" << x.index({0, 2}).item<float>() << '\n';
	// Output:
	// 4.0

	// Extract a single element tensor:
	std::cout << "\"x[1:3]\" as tensor:\n" << x.index({Slice(1, 3), Slice()}) << '\n';
	// Output:
	// tensor [4.0]

	// assign a new vale to an element
	x.index({0, 2}) = -11.0f;
	std::cout << "x = \n" << x << std::endl;

	 //2  1 -11  3
	 //1  2  3  4
	 //4  3  2  1
	 //8  7  3  1

	std::cout << "\"x[:,1:]\":\n" << x.index({Slice(), Slice(1, None)}) << '\n';
	// Output:
	 //1 -11  3
	 //2  3  4
	 //3  2  1
	 //7  3  1
	std::cout << "\"x[:,::2]\":\n" << x.index({Slice(), Slice(None, None, 2)}) << '\n';
	// Output:
	 // 2 -11
	 // 1  3
	 // 4  2
	 // 8  3


	// Combination.
	std::cout << "\"x[:2,1]\":\n" << x.index({Slice(None, 2), 1}) << '\n';
	// Output:
	// 1
	// 2


	// Ellipsis (...).
	std::cout << "\"xs[..., :2]\":\n" << x.index({Ellipsis, Slice(None, 2)}) << "\n\n";
	// Output:
	// 2  1
	// 1  2
	// 4  3
	// 8  7


	std::cout << "--------------- Mutual Transformation of PyTorch and vector-------------------:"  << std::endl;
	// Mutual Transformation of PyTorch and vector


	// NOTE: torch::tensor makes a copy, from_blob does not (but torch::from_blob(vector).clone() does)
	// https://discuss.pytorch.org/t/can-i-initialize-tensor-from-std-vector-in-libtorch/33236/2
	std::array<float, 4> array_data = {1, 2, 3, 4};
	auto tensor_from_array = torch::from_blob(array_data.data(), {2, 2}); // get the pointer ro the data
	std::cout << "Tensor from array:\n" << tensor_from_array << '\n';

	TORCH_CHECK(array_data.data() == tensor_from_array.data_ptr<float>());

	// Tensor from vector:
	std::vector<float> data_vector = {1, 2, 3, 4};
	auto tensor_from_vector = torch::from_blob(data_vector.data(), {2, 2}); // get the pointer ro the data
	std::cout << "Tensor from vector:\n" << tensor_from_vector << "\n\n";

	TORCH_CHECK(data_vector.data() == tensor_from_vector.data_ptr<float>());


	// Tensor From C-style array

	float c_array_data[] = {1, 2, 3, 4};
	auto tensor_from_C_array = torch::from_blob(c_array_data, {2, 2}); // get the pointer ro the data
	std::cout << "Tensor from C - array:\n" << tensor_from_C_array << '\n';

	TORCH_CHECK(c_array_data == tensor_from_C_array.data_ptr<float>());

	return 0;
}


torch::Dimname dimnameFromString(const std::string& str) {
	return torch::Dimname::fromSymbol(torch::Symbol::dimname(str));
}