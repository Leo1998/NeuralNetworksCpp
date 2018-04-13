#pragma once

#include <cmath>
#include <string>

namespace nn {

	static double calcActivation(std::string func, double x)
	{
		if (func == "Sigmoid") {
			double den = 1 + exp(-x);
			return (1 / den);
		}
		else if (func == "Tanh") {
			return tanh(x);
		}
		else if (func == "Linear") {
			return x;
		}
		else if (func == "Relu") {
			return x >= 0.0 ? x : 0.0;
		}
	}

	static double calcDerivative(std::string func, double x)
	{
		if (func == "Sigmoid") {
			double output = calcActivation("Sigmoid", x);

			double derivative = output * (1 - output);
			return derivative;
		}
		else if (func == "Tanh") {
			double t = calcActivation("Tanh", x);
			return 1.0 - t * t;
		}
		else if (func == "Linear") {
			return 1.0;
		}
		else if (func == "Relu") {
			return x >= 0.0 ? 1.0 : 0.0;
		}
	}

}