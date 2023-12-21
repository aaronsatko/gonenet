package activation

// ReLU returns 0 for any negative input, otherwise it returns the input value itself.
//
// Usage:
//   - This function is typically used in hidden layers.
//   - It introduces non-linearity to the model, helping it to learn more complex patterns.
//
// Arguments:
//   - x: A single float64 value representing the neuron's input.
//
// Returns:
//   - A float64 value after applying the ReLU activation function.

func ReLU(x float64) float64 {
	if x < 0 {
		return 0
	}
	return x
}

// ArrRelu applies the ReLU activation function to each element in a slice of float64.
// It is a vectorized version of the ReLU function for handling batches of inputs.
//
// Usage:
//   - This function can be used to apply ReLU activation to a whole layer of neurons in a neural network at once.
//   - Particularly useful when processing inputs in batches, increasing computational efficiency.
//
// Arguments:
//   - input: A slice of float64 values representing the inputs to a layer of neurons.
//
// Returns:
//   - A slice of float64 values where the ReLU function has been applied to each element.

func ArrRelu(input []float64) []float64 {
	output := make([]float64, len(input))
	for i, val := range input {
		output[i] = ReLU(val)
	}
	return output
}
