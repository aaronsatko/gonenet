package layer

import (
	"math/rand"
)

type ActivationFunction func(float64) float64

type DenseLayer struct {
	InputSize  int
	NumNeurons int
	Weights    [][]float64
	Biases     []float64
	Activation ActivationFunction
}

func Dense(inputSize, numNeurons int, activation ActivationFunction) *DenseLayer {
	// init layer with w and b
	weights := make([][]float64, numNeurons)
	for i := range weights {
		weights[i] = make([]float64, inputSize)
		for j := range weights[i] {
			// init with random vals
			weights[i][j] = rand.Float64()
		}
	}

	biases := make([]float64, numNeurons)
	for i := range biases {
		biases[i] = rand.Float64()
	}

	return &DenseLayer{
		InputSize:  inputSize,
		NumNeurons: numNeurons,
		Weights:    weights,
		Biases:     biases,
		Activation: activation,
	}
}

func (dl *DenseLayer) Forward(input []float64) []float64 {
	// forward pass
	output := make([]float64, dl.NumNeurons)

	for i := 0; i < dl.NumNeurons; i++ {
		z := dl.Biases[i]
		for j := 0; j < dl.InputSize; j++ {
			z += dl.Weights[i][j] * input[j]
		}
		output[i] = dl.Activation(z)
	}
	return output
}
