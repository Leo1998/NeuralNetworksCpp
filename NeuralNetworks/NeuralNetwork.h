#pragma once

#include "Matrix.h"
#include "DataSet.h"
#include "Functions.h"

namespace nn {

	class NeuralNetwork
	{
	private:
		int* shape;
		int layerCount;

		Matrix* neurons;
		Matrix* neuronDerivatives;

		ActivationFunction* activationFunctions;
		Matrix* weights;
		Matrix* biases;
	public:
		NeuralNetwork(int shape[], int layerCount, ActivationFunction activationFunctions[]);
		~NeuralNetwork();

		inline int getLayerCount() { return layerCount; }
		inline int getNeuronCount(int layer) { return shape[layer]; }

		inline const Matrix& getNeurons(int l) { return neurons[l]; }
		inline const Matrix& getNeuronDerivatives(int l) { return neuronDerivatives[l]; }

		/**
			Returns the Bias from the l-th Layer to the n-th Neuron in the next Layer.
		*/
		inline double getBias(int l, int n) { return biases[l](n, 0); }
		/**
			Sets the Bias from the l-th Layer to the n-th Neuron in the next Layer to the given Value.
		*/
		inline void setBias(int l, int n, double bias) { biases[l](n, 0) = bias; }

		/**
			Returns the Weight of the Connection from the n1-th Neuron in the l-th Layer to the	n2-th Neuron in the next Layer.
		*/
		inline double getWeight(int l, int n1, int n2) { return weights[l](n1, n2); }
		/**
			Sets the Weight of the Connection from the n1-th Neuron in the l-th Layer to the n2-th Neuron in the next Layer to the given Value.
		*/
		inline void setWeight(int l, int n1, int n2, double value) { weights[l](n1, n2) = value; }

		inline const Matrix& getWeightMatrix(int l) { return weights[l]; }
		inline const Matrix& getBiasMatrix(int l) { return biases[l]; }

		inline void adjustWeights(int l, const Matrix& delta) { weights[l] += delta; }
		inline void adjustBiases(int l, const Matrix& delta) { biases[l] += delta; }

		void initializeRandom(double min, double max);
		/**
			uniform: Whether to use uniform or normal distributed random initialization.
		*/
		void initializeXavier();

		void initializeMinibatchSize(int minibatchSize);
		Matrix* compute(const DataSet& data, bool calcDerivatives = false);

	};
}
