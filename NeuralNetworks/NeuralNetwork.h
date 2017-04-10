#pragma once

#include <vector>

#include "Layer.h"

class NeuralNetwork {
private:
	std::vector<Layer> layers;
public:
	NeuralNetwork(std::vector<int> pattern);
	~NeuralNetwork();

	void logSchema();
};