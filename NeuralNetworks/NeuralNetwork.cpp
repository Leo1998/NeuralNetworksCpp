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

Matrix* NeuralNetwork::compute(const Matrix& input) {
	Matrix* m = new Matrix(input.getRows(), input.getCols() + 1);
	copyMatrixAndAddBias(input, *m);

	for (int l = 0; l < layerCount - 1; l++) {
		Matrix temp = *m * weights[l];

		bool needBias = l != layerCount - 2;
		*m = Matrix(temp.getRows(), needBias ? temp.getCols() + 1 : temp.getCols());
		copyMatrixAndAddBias(temp, *m);

		for (int i = 0; i < m->getRows(); i++) {
			for (int j = 0; j < m->getCols(); j++) {
				m->operator()(i, j) = activationFunction->getOutput(m->operator()(i, j));
			}
		}
	}

	return m;
}
