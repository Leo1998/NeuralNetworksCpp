#include "Optimizer.h"

#include "MathUtil.h"

namespace nn {

	Optimizer::Optimizer(NeuralNetwork* nn) : nn(nn)
	{
	}


	Optimizer::~Optimizer()
	{
		delete[] errors;
		delete[] weightDeltas;
		delete[] biasDeltas;

		if (lastWeightDeltas) {
			delete[] lastWeightDeltas;
			delete[] lastBiasDeltas;
		}
	}

	void Optimizer::initializeMinibatchSize(int minibatchSize)
	{
		if (errors == nullptr || (errors != nullptr && errors[0].getRows() != minibatchSize)) {
			delete[] errors;
			delete[] weightDeltas;
			delete[] biasDeltas;

			if (lastWeightDeltas) {
				delete[] lastWeightDeltas;
				delete[] lastBiasDeltas;
			}

			this->errors = new Matrix[nn->getLayerCount()];
			for (int l = 0; l < nn->getLayerCount(); l++) {
				errors[l] = Matrix(nn->getNeuronCount(l), minibatchSize);
			}

			this->weightDeltas = new Matrix[nn->getLayerCount() - 1];
			for (int l = 0; l < nn->getLayerCount() - 1; l++) {
				weightDeltas[l] = Matrix(nn->getNeuronCount(l), nn->getNeuronCount(l + 1));
			}
			this->biasDeltas = new Matrix[nn->getLayerCount() - 1];
			for (int l = 0; l < nn->getLayerCount() - 1; l++) {
				biasDeltas[l] = Matrix(nn->getNeuronCount(l + 1), 1);
			}

			this->lastWeightDeltas = new Matrix[nn->getLayerCount() - 1];
			for (int l = 0; l < nn->getLayerCount() - 1; l++) {
				lastWeightDeltas[l] = Matrix(nn->getNeuronCount(l), nn->getNeuronCount(l + 1));
			}
			this->lastBiasDeltas = new Matrix[nn->getLayerCount() - 1];
			for (int l = 0; l < nn->getLayerCount() - 1; l++) {
				lastBiasDeltas[l] = Matrix(nn->getNeuronCount(l + 1), 1);
			}
		}
	}

	void Optimizer::optimize(DataSet data, double learningRate, double momentum)
	{
		initializeMinibatchSize(data.getMinibatchSize());

		Matrix* output = nn->compute(data, true);

		for (int l = nn->getLayerCount() - 1; l > 0; l--) {
			int next = l + 1;

			if (l == nn->getLayerCount() - 1) {
				errors[l] = data.getDesiredOutput() - *output; // this decides if we need to add or subtract the delta weights
				errors[l] = errors[l].transpose();
			}
			else {
				errors[l] = nn->getWeightMatrix(l) * errors[l + 1];

				const Matrix& derivatives = nn->getNeuronDerivatives(l);
				errors[l].multElementWise(derivatives.transpose());
			}
		}

		for (int l = 0; l < nn->getLayerCount() - 1; l++) {
			weightDeltas[l] = (errors[l + 1] * nn->getNeurons(l)).transpose();

			for (int i = 0; i < errors[l + 1].getRows(); i++) {
				double sum = 0.0;
				for (int j = 0; j < errors[l + 1].getCols(); j++) {
					sum += errors[l + 1](i, j);
				}
				biasDeltas[l](i, 0) = sum;
			}
		}

		for (int l = 0; l < nn->getLayerCount() - 1; l++) {
			Matrix dw = (learningRate * weightDeltas[l]) + (momentum * lastWeightDeltas[l]);
			Matrix db = (learningRate * biasDeltas[l]) + (momentum * lastBiasDeltas[l]);
			nn->adjustWeights(l, dw);
			nn->adjustBiases(l, db);

			lastWeightDeltas[l] = dw;
			lastBiasDeltas[l] = db;
		}
	}

	double Optimizer::calcLoss(DataSet data)
	{
		Matrix* actualOutput = nn->compute(data);

		Matrix delta = *actualOutput - data.getDesiredOutput();
		delta.multElementWise(delta);
		double totalError = delta.sum();

		return sqrt(totalError / (actualOutput->getRows() * actualOutput->getCols()));
	}
}