#include "Optimizer.h"

Optimizer::Optimizer(NeuralNetwork* nn) : nn(nn)
{
}


Optimizer::~Optimizer()
{
	delete[] errors;
	delete[] weightDeltas;
}

void Optimizer::initializeMinibatchSize(int minibatchSize)
{
	if (errors == nullptr || (errors != nullptr && errors[0].getRows() != minibatchSize)) {
		delete[] errors;
		delete[] weightDeltas;

		this->errors = new Matrix[nn->getLayerCount()];
		for (int l = 0; l < nn->getLayerCount(); l++) {
			errors[l] = Matrix(minibatchSize, nn->getNeuronCount(l));
		}

		this->weightDeltas = new Matrix[nn->getLayerCount() - 1];
		for (int l = 0; l < nn->getLayerCount() - 1; l++) {
			weightDeltas[l] = Matrix(nn->getNeuronCount(l), nn->getNeuronCount(l + 1));
		}

		this->biasDeltas = new Matrix[nn->getLayerCount() - 1];
		for (int l = 0; l < nn->getLayerCount() - 1; l++) {
			biasDeltas[l] = Matrix(nn->getNeuronCount(l + 1), 1);
		}
	}
}

void Optimizer::optimize(DataSet data, double learningRate, double momentum)
{
	if (data.hasDesiredOutput()) {
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

			Matrix one = Matrix(data.getMinibatchSize(), nn->getNeuronCount(l));
			one.fill(1.0);
			biasDeltas[l] = (errors[l + 1] * one).transpose();

			Matrix dw = learningRate * weightDeltas[l];
			Matrix db = learningRate * biasDeltas[l];
			nn->adjustWeights(l, dw);
			nn->adjustBiases(l, db);
		}
	}
}
