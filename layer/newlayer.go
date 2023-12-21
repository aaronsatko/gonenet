package layer

import (
	"math/rand"
)

// Defines the type for activation function.
type ActivationFunction func(float64) float64

// Layer is a base type that defines common properties of all layer types in a neural network.
//
// Fields:
//   - InputSize: Number of inputs the layer receives. It depends on the size of the previous layer or the feature space.
//   - NumNeurons: Number of neurons (or units) in the layer.
//   - Weights: A 2D slice of float64 representing the weights associated with each neuron and input.
//   - Biases: A slice of float64 representing the bias for each neuron.
//   - Activation: The activation function used by the neurons in the layer.
//
// Usage:
//   - This base type is used to create various specific types of layers like dense, convolutional, recurrent, etc.
//   - It encapsulates the common functionalities and properties needed for different layer types.

type Layer struct {
	InputSize  int
	NumNeurons int
	Weights    [][]float64
	Biases     []float64
	Activation ActivationFunction
}

// NewLayer is a generic function for creating a new layer with specified properties.
// Initializes a layer with random weights and biases.
//
// Arguments:
//   - inputSize: The number of inputs the layer will receive.
//   - numNeurons: The number of neurons in the layer.
//   - activation: The activation function to be used in the layer.
//
// Returns:
//   - A pointer to an initialized Layer instance.
//
// Note:
//   - Weights and biases are initialized with random values for start.
//   - This function provides a general-purpose way to create layers of different types.

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

// Forward performs the forward pass for the layer.
// It computes the output of the layer given an input.
//
// Arguments:
//   - input: A slice of float64 representing the input to the layer.
//
// Returns:
//   - A slice of float64 representing the output from the layer after applying the weights, biases, and activation function.
//
// Note:
//   - Calculates the linear combination of inputs and weights, adds the bias, and applies the activation function.

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
