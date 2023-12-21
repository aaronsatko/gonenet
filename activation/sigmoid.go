package activation

import (
	"math"
)

// Sigmoid is an activation function that maps any real value into the range (0, 1).
// It is commonly used in binary classification problems.
//
// Usage:
//   - Often used in the output layer of a binary classification neural network.
//   - It helps to transform outputs into probabilities, as its output is always between 0 and 1.
//
// Arguments:
//   - X: A single float64 value representing the neuron's input.
//
// Returns:
//   - A float64 value after applying the Sigmoid activation function.
//     The return value is always in the range (0, 1).
//
// Note:
//   - While Sigmoid is useful in certain scenarios, it can suffer from the vanishing gradient problem
//     during backpropagation, especially with deep networks. In such cases, alternative functions like
//     ReLU would usually be preferred.

func Sigmoid(X float64) float64 {
	return 1 / (1 + math.Exp(-X))
}

// ArrSigmoid applies the Sigmoid activation function to each element in a slice of float64.

func ArrSigmoid(input []float64) []float64 {
	output := make([]float64, len(input))
	for i, val := range input {
		output[i] = Sigmoid(val)
	}
	return output
}
