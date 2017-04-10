#pragma once

#include <vector>

#include "Neuron.h"

class Layer {
private:
	std::vector<Neuron> neurons;
public:
	Layer(int neuronCount);
	~Layer();

	int getNeuronCount();
};

