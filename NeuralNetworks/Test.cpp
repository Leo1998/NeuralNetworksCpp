#include "NeuralNetwork.h"
#include "ActivationFunctions.h"

#include "Timer.h"
#include <iostream>

int main() {
	int shape[] = { 2, 2, 1 };

	NeuralNetwork nn(shape, sizeof(shape) / sizeof(*shape), new Sigmoid);
	nn.randomizeWeights(0.0, 1.0);

	Matrix m(2, 2);
	m(0, 0) = 1.0;
	m(0, 1) = 0.0;

	m(1, 0) = 0.0;
	m(1, 1) = 1.0;

	Timer timer;
	Matrix* result = nn.compute(m);

	std::cout << "Computation took: " << timer.elapsedMillis() << " ms" << std::endl;
	std::cout << *result << std::endl;

	std::cin.get();
	
	return 0;
}