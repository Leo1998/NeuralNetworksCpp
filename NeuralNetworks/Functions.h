#pragma once

#include <cmath>

namespace nn {

	static void applyLinear(Matrix& in) {
		//nothin
	}

	static void applyLinearDerivative(Matrix& in) {
		in.fill(1.0);
	}
	
	static void applySigmoid(Matrix& in) {
		for (int i = 0; i < in.getRows(); i++) {
			for (int j = 0; j < in.getCols(); j++) {
				in(i, j) = 1 / (1 + exp(-in(i, j)));
			}
		}
	}

	static void applySigmoidDerivative(Matrix& in) {
		applySigmoid(in);
		for (int i = 0; i < in.getRows(); i++) {
			for (int j = 0; j < in.getCols(); j++) {
				double x = in(i, j);

				in(i, j) = x * (1 - x);
			}
		}
	}

	static void applyTanh(Matrix& in) {
		for (int i = 0; i < in.getRows(); i++) {
			for (int j = 0; j < in.getCols(); j++) {
				in(i, j) = tanh(in(i, j));
			}
		}
	}

	static void applyTanhDerivative(Matrix& in) {
		applyTanh(in);
		for (int i = 0; i < in.getRows(); i++) {
			for (int j = 0; j < in.getCols(); j++) {
				double x = in(i, j);
				
				in(i, j) = 1.0 - x * x;
			}
		}
	}

	static void applyRelu(Matrix& in) {
		for (int i = 0; i < in.getRows(); i++) {
			for (int j = 0; j < in.getCols(); j++) {
				in(i, j) = in(i, j) >= 0.0 ? in(i, j) : 0.0;
			}
		}
	}

	static void applyReluDerivative(Matrix& in) {
		for (int i = 0; i < in.getRows(); i++) {
			for (int j = 0; j < in.getCols(); j++) {
				in(i, j) = in(i, j) >= 0.0 ? 1.0 : 0.0;
			}
		}
	}

	static void applySoftmax(Matrix& in) {
		for (int i = 0; i < in.getRows(); i++) {
			double sumExp = 0.0;
			for (int j = 0; j < in.getCols(); j++) {
				sumExp += exp(in(i, j));
			}

			for (int j = 0; j < in.getCols(); j++) {
				in(i, j) = exp(in(i, j)) / sumExp;
			}
		}
	}

	static void applySoftmaxDerivative(Matrix& in) {
		for (int i = 0; i < in.getRows(); i++) {
			for (int j = 0; j < in.getCols(); j++) {
				in(i, j) = 0.0; //TODO
			}
		}
	}

	static int kroneckerDelta(int i, int j) {
		return i == j ? 1 : 0;
	}

	enum ActivationFunction {
		Linear,
		Sigmoid,
		Tanh,
		Relu,
		Softmax
	};

	typedef void(*func)(Matrix& in);

	static func getFunction(ActivationFunction func) {
		if (func == Sigmoid) {
			return applySigmoid;
		}
		else if (func == Tanh) {
			return applyTanh;
		}
		else if (func == Relu) {
			return applyRelu;
		}
		else if (func == Softmax) {
			return applySoftmax;
		}

		return applyLinear;
	}

	static func getDerivative(ActivationFunction func) {
		if (func == Sigmoid) {
			return applySigmoidDerivative;
		}
		else if (func == Tanh) {
			return applyTanhDerivative;
		}
		else if (func == Relu) {
			return applyReluDerivative;
		}
		else if (func == Softmax) {
			return applySoftmaxDerivative;
		}

		return applyLinearDerivative;
	}

	/*static double calcActivation(std::string func, double x)
	{

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
	}*/

}