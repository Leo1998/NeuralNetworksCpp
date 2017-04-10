#include "Log.h"
#include "NeuralNetwork.h"

int main() {
	NeuralNetwork n({2, 3, 2});

	n.logSchema();

	waitHere();

	return 0;
}