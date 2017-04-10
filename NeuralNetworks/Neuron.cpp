#include "Neuron.h"

Neuron::Neuron()
{
	this->inputFunction = new SumInput();
	this->transferFunction = new LinearTransfer();
}

Neuron::Neuron(InputFunction* inputFunction, TransferFunction* transferFunction)
{
	this->inputFunction = inputFunction;
	this->transferFunction = transferFunction;
}

Neuron::~Neuron()
{
}

void Neuron::calculate() {
	totalInput = inputFunction->getOutput();
	output = transferFunction->getOutput(totalInput);
}