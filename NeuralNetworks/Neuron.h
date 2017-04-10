#pragma once

#include "Functions.h"

class Neuron
{
private:
	InputFunction* inputFunction;
	TransferFunction* transferFunction;

	double totalInput;
	double output;
public:
	Neuron();
	Neuron(InputFunction* inputFunction, TransferFunction* transferFunction);
	~Neuron();

	virtual void calculate();
};