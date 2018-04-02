#pragma once

#include "NeuralNetwork.h"

class Optimizer
{
private:
	NeuralNetwork* nn;
	Matrix* errors;
	Matrix* errorDeltas;
public:
	Optimizer(NeuralNetwork* nn);
	~Optimizer();

	void initializeMinibatchSize(int minibatchSize);
	void optimize(DataSet data, double learningRate, double momentum);
};

