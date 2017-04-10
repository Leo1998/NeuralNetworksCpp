#pragma once

// input functions

class InputFunction
{
public:
	virtual double getOutput() = 0;
};

class SumInput : public InputFunction
{
	double getOutput() {
		return 0.0;//TODO
	}
};

// transfer functions

class TransferFunction
{
public:
	virtual double getOutput(double totalInput) = 0;

	virtual double getDerivative(double totalInput) {
		return 1.0;
	}
};

class LinearTransfer : public TransferFunction
{
	double getOutput(double totalInput) {
		return totalInput;
	}

	double getDerivative(double totalInput) {
		return 1.0;
	}
};