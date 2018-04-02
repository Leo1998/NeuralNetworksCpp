#include "DataSet.h"

DataSet::DataSet(Matrix& input)
{
	this->input = &input;
}

DataSet::DataSet(Matrix& input, Matrix& desiredOutput)
{
	this->input = &input;
	this->desiredOutput = &desiredOutput;
}
