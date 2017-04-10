#include "Layer.h"

Layer::Layer(int neuronCount)
{
	for (int i = 0; i < neuronCount; i++) {
		Neuron n;

		neurons.push_back(n);
	}
}

Layer::~Layer()
{
}

int Layer::getNeuronCount() {
	return neurons.size();
}
