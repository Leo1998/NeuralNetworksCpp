#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(int shape[], int layerCount, ActivationFunction* activationFunction) : layerCount(layerCount), activationFunction(activationFunction)
{

	this->shape = new int[layerCount];
	memcpy(this->shape, shape, layerCount * sizeof(int));

	this->weights = new Matrix[layerCount - 1];
	for (int l = 0; l < layerCount - 1; l++) {
		weights[l] = Matrix(getNeuronCount(l) + 1, getNeuronCount(l + 1));
	}
}

NeuralNetwork::~NeuralNetwork()
{
	delete[] shape;
	delete[] weights;
	delete activationFunction;
}

void NeuralNetwork::randomizeWeights(double min, double max) {
	for (int l = 0; l < getLayerCount() - 1; l++) {
		weights[l].randomize(min, max);
	}
}

static void copyMatrixAndAddBias(const Matrix& src, const Matrix& dest)
{
	for (int i = 0; i < dest.getRows(); i++) {
		for (int j = 0; j < dest.getCols(); j++) {
			if (j == src.getCols()) {
				dest(i, j) = 1.0;
			}
			else {
				dest(i, j) = src(i, j);
			}
		}
	}
}

void NeuralNetwork::initializeMinibatchSize(int minibatchSize)
{
	if (neurons == nullptr || (neurons != nullptr && neurons[0].getRows() != minibatchSize)) {
		delete[] neurons;
		delete[] neuronDerivatives;
		
		this->neurons = new Matrix[layerCount];
		for (int l = 0; l < layerCount; l++) {
			neurons[l] = Matrix(minibatchSize, l == layerCount - 1 ? getNeuronCount(l) : getNeuronCount(l) + 1);
		}

		this->neuronDerivatives = new Matrix[layerCount];
		for (int l = 0; l < layerCount; l++) {
			neuronDerivatives[l] = Matrix(minibatchSize, getNeuronCount(l));
		}
	}
}

Matrix* NeuralNetwork::compute(const DataSet& data, bool calcDerivatives) {
	initializeMinibatchSize(data.getInput().getRows());

	copyMatrixAndAddBias(data.getInput(), neurons[0]);

	for (int l = 0; l < layerCount - 1; l++) {
		Matrix* in = &(neurons[l]);
		Matrix m = *in * weights[l];

		for (int i = 0; i < m.getRows(); i++) {
			for (int j = 0; j < m.getCols(); j++) {
				m(i, j) = activationFunction->getOutput(m(i, j));

				if (calcDerivatives)
					neuronDerivatives[l](i, j) = activationFunction->getDerivative(m(i, j));
			}
		}

		copyMatrixAndAddBias(m, neurons[l + 1]);
	}

	return &(neurons[getLayerCount() - 1]);
}
