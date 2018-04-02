#include "Optimizer.h"



Optimizer::Optimizer(NeuralNetwork* nn) : nn(nn)
{
}


Optimizer::~Optimizer()
{
	delete[] errors;
	delete[] errorDeltas;
}

void Optimizer::initializeMinibatchSize(int minibatchSize)
{
	if (errors == nullptr || (errors != nullptr && errors[0].getRows() != minibatchSize)) {
		delete[] errors;
		delete[] errorDeltas;

		this->errors = new Matrix[nn->getLayerCount()];
		for (int l = 0; l < nn->getLayerCount(); l++) {
			errors[l] = Matrix(minibatchSize, nn->getNeuronCount(l));
		}

		this->errorDeltas = new Matrix[nn->getLayerCount()];
		for (int l = 0; l < nn->getLayerCount(); l++) {
			errorDeltas[l] = Matrix(minibatchSize, nn->getNeuronCount(l));
		}
	}
}

void Optimizer::optimize(DataSet data, double learningRate, double momentum)
{
	if (data.hasDesiredOutput()) {
		initializeMinibatchSize(data.getMinibatchSize());

		Matrix* output = nn->compute(data, true);

		for (int l = nn->getLayerCount() - 1; l >= 0; l--) {
			int next = l + 1;

			if (l == nn->getLayerCount()) {
				errors[l] = data.getDesiredOutput() - *output;
				errorDeltas[l] = errors[l] * nn->getNeuronDerivatives(l);
			}



			/*for (int i = 0; i < nn->getNeuronCount(l); i++) {
				if (l == nn->getLayerCount() - 1) {
					errorMatrix[l].set(i, 0, desiredOutput[i] - output[i]);
				}
				else {
					for (int j = 0; j < nn.countNeurons(next); j++) {
						accumulateDelta(l, i, j, errorDeltaMatrix[next].get(j, 0) * nn.getNeuron(l, i));

						errorMatrix[l].add(i, 0, nn.getWeight(l, i, j) * errorDeltaMatrix[next].get(j, 0));
					}
				}
				errorDeltaMatrix[l].set(i, 0, errorMatrix[l].get(i, 0) * nn.getTransferFunction().getDerivative(nn.getNeuron(l, i)));
			}

			if (l < nn.getLayerCount() - 1) {
				for (int j = 0; j < nn.countNeurons(next); j++) {
					accumulateBiasDelta(l, j, errorDeltaMatrix[next].get(j, 0));
				}
			}*/
		}
	}
}
