package layer

import (
	"math/rand"
)

type ActivationFunction func(float64) float64

// Layer is a base type that defines common properties of all layer types
type Layer struct {
	InputSize  int
	NumNeurons int
	Weights    [][]float64
	Biases     []float64
	Activation ActivationFunction
}

// NewLayer is a generic function for creating layers
func NewLayer(inputSize, numNeurons int, activation ActivationFunction) *Layer {
	// Initialize layer with weights and biases
	weights := make([][]float64, numNeurons)
	for i := range weights {
		weights[i] = make([]float64, inputSize)
		for j := range weights[i] {
			// Initialize with random values
			weights[i][j] = rand.Float64()
		}
	}

	biases := make([]float64, numNeurons)
	for i := range biases {
		biases[i] = rand.Float64()
	}

	return &Layer{
		InputSize:  inputSize,
		NumNeurons: numNeurons,
		Weights:    weights,
		Biases:     biases,
		Activation: activation,
	}
}

func (l *Layer) Forward(input []float64) []float64 {
	// Forward pass
	output := make([]float64, l.NumNeurons)

	for i := 0; i < l.NumNeurons; i++ {
		z := l.Biases[i]
		for j := 0; j < l.InputSize; j++ {
			z += l.Weights[i][j] * input[j]
		}
		output[i] = l.Activation(z)
	}
	return output
}
