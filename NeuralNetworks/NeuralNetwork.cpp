#include "NeuralNetwork.h"

namespace nn {

	NeuralNetwork::NeuralNetwork(int shape[], int layerCount, std::string activationFunctions[]) : layerCount(layerCount), activationFunctions(activationFunctions)
	{
		this->shape = new int[layerCount];
		memcpy(this->shape, shape, layerCount * sizeof(int));

		this->weights = new Matrix[layerCount - 1];
		for (int l = 0; l < layerCount - 1; l++) {
			weights[l] = Matrix(getNeuronCount(l), getNeuronCount(l + 1));
		}

		this->biases = new Matrix[layerCount - 1];
		for (int l = 0; l < layerCount - 1; l++) {
			biases[l] = Matrix(getNeuronCount(l + 1), 1);
		}
	}

	NeuralNetwork::~NeuralNetwork()
	{
		delete[] shape;
		delete[] weights;
		delete[] activationFunctions;
	}

	void NeuralNetwork::randomizeWeights(double min, double max) {
		for (int l = 0; l < getLayerCount() - 1; l++) {
			weights[l].randomize(min, max);
			biases[l].randomize(min, max);
		}
	}

	void NeuralNetwork::initializeMinibatchSize(int minibatchSize)
	{
		if (neurons == nullptr || (neurons != nullptr && neurons[0].getRows() != minibatchSize)) {
			delete[] neurons;
			delete[] neuronDerivatives;

			this->neurons = new Matrix[layerCount];
			for (int l = 0; l < layerCount; l++) {
				neurons[l] = Matrix(minibatchSize, getNeuronCount(l));
			}

			this->neuronDerivatives = new Matrix[layerCount];
			for (int l = 0; l < layerCount; l++) {
				neuronDerivatives[l] = Matrix(minibatchSize, getNeuronCount(l));
			}
		}
	}

	Matrix* NeuralNetwork::compute(const DataSet& data, bool calcDerivatives) {
		initializeMinibatchSize(data.getInput().getRows());

		neurons[0] = data.getInput();

		for (int l = 0; l < layerCount - 1; l++) {
			Matrix* in = &(neurons[l]);
			neurons[l + 1] = *in * weights[l];

			for (int i = 0; i < neurons[l + 1].getRows(); i++) {
				for (int j = 0; j < neurons[l + 1].getCols(); j++) {
					// add bias
					neurons[l + 1](i, j) += biases[l](j, 0);

					if (calcDerivatives)
						neuronDerivatives[l + 1](i, j) = calcDerivative(activationFunctions[l + 1], neurons[l + 1](i, j));

					neurons[l + 1](i, j) = calcActivation(activationFunctions[l + 1], neurons[l + 1](i, j));
				}
			}
		}

		return &(neurons[getLayerCount() - 1]);
	}
}