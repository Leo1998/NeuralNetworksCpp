#include "NeuralNetwork.h"
#include "Optimizer.h"

#include "Timer.h"
#include <iostream>

using namespace nn;

int main0() {
	int shape[] = { 2, 2, 1 };
	ActivationFunction activation[] = {Linear, Tanh, Tanh};

	NeuralNetwork nn(shape, sizeof(shape) / sizeof(*shape), activation);
	nn.initializeRandom(-0.5, 0.5);

	Matrix input(4, 2);
	input(0, 0) = 0.0;
	input(0, 1) = 0.0;
	input(1, 0) = 0.0;
	input(1, 1) = 1.0;
	input(2, 0) = 1.0;
	input(2, 1) = 0.0;
	input(3, 0) = 1.0;
	input(3, 1) = 1.0;

	Matrix desiredOutput(4, 1);
	desiredOutput(0, 0) = 0.0;
	desiredOutput(1, 0) = 1.0;
	desiredOutput(2, 0) = 1.0;
	desiredOutput(3, 0) = 0.0;
	DataSet dataSet(input, desiredOutput);

	Optimizer optimizer(&nn);

	for (int i = 0; i < 5000; i++) {
		optimizer.optimize(dataSet, 0.049, 0.1);
		std::cout << optimizer.calcLoss(dataSet) << std::endl;
	}

	Timer timer;
	Matrix* result = nn.compute(input);

	std::cout << "Computation took: " << timer.elapsedMillis() << " ms" << std::endl;
	std::cout << *result << std::endl;

	std::cin.get();
	
	return 0;
}