#pragma once

#include <cmath>

class ActivationFunction
{
public:
	virtual double getOutput(double x) = 0;

	virtual double getDerivative(double x) = 0;
};

class Linear : public ActivationFunction
{
	double getOutput(double x) {
		return x;
	}

	double getDerivative(double x) {
		return 1.0;
	}
};

class Sigmoid : public ActivationFunction
{
	double getOutput(double x)
	{
		if (x > 100) {
			return 1.0;
		}
		else if (x < -100) {
			return 0.0;
		}

		double den = 1 + exp(-x);
		return (1 / den);
	}

	double getDerivative(double x)
	{
		double output = getOutput(x);

		double derivative = output * (1 - output);
		return derivative;
	}
};