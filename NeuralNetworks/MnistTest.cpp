#include "NeuralNetwork.h"
#include "Functions.h"
#include "Optimizer.h"

#include "mnist_reader.h"
#include <vector>

#include "Timer.h"

#include <iostream>
#include <string>

#define BATCH_SIZE 50
#define TRAIN_SIZE 10000

using namespace nn;

int main() {

	mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset = 
		mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(TRAIN_SIZE, 1);

	std::cout << "Database loaded!" << std::endl;

	DataSet* batches = new DataSet[TRAIN_SIZE / BATCH_SIZE];
	for (int b = 0; b < TRAIN_SIZE; b += BATCH_SIZE) {
		Matrix in(BATCH_SIZE, 784);
		Matrix out(BATCH_SIZE, 10);

		for (int i = 0; i < BATCH_SIZE; i++) {
			std::vector<uint8_t> image = dataset.training_images[b + i];
			int label = (int)dataset.training_labels[b + i];

			for (int p = 0; p < image.size(); p++) {
				in(i, p) = (int)image[p] / 255.0;
			}
			out(i, label) = 1.0;
		}
		batches[b / BATCH_SIZE] = DataSet(in, out);
	}

	std::cout << "Training Data prepared!" << std::endl;
	
	int shape[] = { 784, 800, 10 };
	NeuralNetwork nn(shape, sizeof(shape) / sizeof(*shape), new std::string[3]{ "", "Tanh", "Tanh" });
	nn.randomizeWeights(-1.0, 1.0);

	Optimizer optimizer(&nn);
	
	for (int e = 0; e < 10; e++) {
		for (int b = 0; b < TRAIN_SIZE; b += BATCH_SIZE) {
			DataSet dataset = batches[b / BATCH_SIZE];

			optimizer.optimize(dataset, 0.01, 0.9);

			std::cout << "Epoch: " << e << " Batch: " << b / BATCH_SIZE << " Loss: " << optimizer.calcLoss(dataset) << std::endl;
		}

		
	}

	std::cout << "Testing..." << std::endl;
	int correctCount = 0;
	for (int b = 0; b < TRAIN_SIZE; b += BATCH_SIZE) {
		DataSet dataset = batches[b / BATCH_SIZE];
		Matrix* output = nn.compute(dataset);

		for (int i = 0; i < output->getRows(); i++) {
			int max = (*output)(i, 0);
			int maxIdx = 0;
			for (int j = 1; j < output->getCols(); j++) {
				if ((*output)(i, j) > max) {
					max = (*output)(i, j);
					maxIdx = j;
				}
			}

			if (dataset.getDesiredOutput()(i, maxIdx) > 0) {
				correctCount++;
			}
		}
	}
	std::cout << correctCount << " of " << TRAIN_SIZE << " correct!" << std::endl;

	delete[] batches;

	std::cin.get();

	return 0;
}