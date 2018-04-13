#include "DataSet.h"

namespace nn {

	DataSet::DataSet()
	{

	}

	DataSet::DataSet(const Matrix& input)
	{
		this->input = input;
	}

	DataSet::DataSet(const Matrix& input, const Matrix& desiredOutput)
	{
		this->input = input;
		this->desiredOutput = desiredOutput;
	}
}