#pragma once

#include "NeuralNetwork.h"

namespace nn {

	class Optimizer
	{
	private:
		NeuralNetwork * nn;
		Matrix* errors;
		Matrix* weightDeltas;
		Matrix* biasDeltas;
		Matrix* lastWeightDeltas;
		Matrix* lastBiasDeltas;
	public:
		Optimizer(NeuralNetwork* nn);
		~Optimizer();

		void initializeMinibatchSize(int minibatchSize);
		void optimize(DataSet data, double learningRate, double momentum);
		double calcLoss(DataSet data);
	};
}
