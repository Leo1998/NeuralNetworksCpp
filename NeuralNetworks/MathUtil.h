#pragma once

#include <cstdlib>
#include <ctime>

static bool seedSet = false;

static double randomDouble(double min, double max)
{
	if (!seedSet) {
		srand((unsigned int)time(NULL));
		seedSet = true;
	}

	double f = (double) rand() / RAND_MAX;
	return min + f * (max - min);
}