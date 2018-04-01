#pragma once

#include <cstdlib>

static double randomDouble(double min, double max)
{
	double f = (double) rand() / RAND_MAX;
	return min + f * (max - min);
}