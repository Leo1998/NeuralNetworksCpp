#pragma once

#include "Matrix.h"

namespace nn {

	class DataSet
	{
	private:
		Matrix input;
		Matrix desiredOutput;
	public:
		DataSet();
		DataSet(const Matrix& input);
		DataSet(const Matrix& input, const Matrix& desiredOutput);

		inline const Matrix& getInput() const { return input; }
		inline const Matrix& getDesiredOutput() const { return desiredOutput; }

		inline int getMinibatchSize() { return input.getRows(); }

	};
}
