#include <torch/torch.h>
#include <torch/script.h>
#include <array>
#include <vector>
#include <iostream>
#include <iomanip>

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

int main()
{

	std::cout << "-----------------------------------------------------------------------"  << std::endl;
	std::cout << "--------------------Basic probability theory---------------------------"  << std::endl;
	std::cout << "----------------	  Getting Started: "  									<< std::endl;

	std::cout << std::fixed << std::setprecision(4);


	// Next, we'll want to be able to cast the die. In statistics we call this process of drawing examples from probability distributions sampling.
	// The distribution which assigns probabilities to a number of discrete choices is called the multinomial distribution.
	// We'll give a more formal definition of distribution later, but at a high level, think of it as just an assignment of probabilities to events

	auto options =
	    torch::TensorOptions()
	    .dtype(torch::kFloat32);

	// In PyTorch, we can sample from the multinomial distribution via the aptly named
	// 						torch::multinomial function
	// To draw a single sample, we simply pass in a vector of probabilities.

	auto probabilities = torch::ones({6}, options) / 6;
	std::cout << "Random Sample: \n" << probabilities.multinomial(/*number of samples*/1, /*replacement*/ false).item<int>() << std::endl;

	std::cout << "--------------------Rolling a dice ---------------------------"  << std::endl;
	auto num_rolls = 1000;
	auto totals = torch::zeros(6, options);
	int get_index;

	for (int i = 0; i < num_rolls; ++i) {
		get_index = probabilities.multinomial(/*number of samples*/1, /*replacement*/ false).item<int>();
		totals.index({get_index}) += 1;
	}

	std::cout << totals / num_rolls << std::endl;
	std::cout << "--------------------Integer Random numbers  ---------------------------"  << std::endl;
	std::cout << torch::randint(10, 20, torch::TensorOptions().dtype(torch::kUInt8)) << std::endl;
	std::cout << "--------------------Normal Random numbers  ---------------------------"  << std::endl;
	std::cout << torch::randn(10) << std::endl;
	return 0;
}
