#include "NeuralNetwork.h"

#include "Log.h"

NeuralNetwork::NeuralNetwork(std::vector<int> pattern) 
{
	for (int c : pattern) {
		Layer l(c);

		layers.push_back(l);
	}
}

NeuralNetwork::~NeuralNetwork() 
{
}

void NeuralNetwork::logSchema() {
	LOG_INFO("NeuralNetwork Schema:");

	int i = 0;
	for (Layer l : layers) {
		LOG_INFO("Layer: ", i, " Neurons: ", l.getNeuronCount());

		i++;
	}
}