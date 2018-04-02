#pragma once

#include "Matrix.h"

class DataSet
{
private:
	Matrix* input;
	Matrix* desiredOutput;
public:
	DataSet(Matrix& input);
	DataSet(Matrix& input, Matrix& desiredOutput);

	inline const Matrix& getInput() const { return *input; }
	inline const Matrix& getDesiredOutput() const { return *desiredOutput; }

	inline int getMinibatchSize() { return input->getRows(); }
	inline bool hasDesiredOutput() { return desiredOutput; }

};

