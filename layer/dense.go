package layer

// DenseLayer represents a fully connected (dense) layer in a neural network.
// In a dense layer, each neuron receives input from all neurons in the previous layer.
//
// Fields:
//   - Layer: A pointer to the base Layer struct (includes fields like weights, biases, and activation function).
//
// Usage:
//   - Dense layers are versatile and can be used in almost any neural network architecture,
//     including deep learning for tasks like classification and regression.
//
// Note:
//   - The dense layer is a crucial building block in neural networks and is often used in both the hidden layers
//     and the output layer of a network.

type DenseLayer struct {
	*Layer
}

// Dense creates a new instance of a dense (fully connected) layer with the specified parameters.
// It initializes a layer with a given number of neurons and an activation function.
//
// Arguments:
//   - inputSize: Number of input units to the layer (depends on the size of the previous layer or the feature space).
//   - numNeurons: Number of neurons in the dense layer.
//   - activation: Activation function to be used in the layer.
//
// Returns:
//   - A pointer to an initialized DenseLayer instance.
//
// Note:
//   - The activation function is applied to the linear combination of weights and inputs of the layer.
//   - This function sets up the initial structure of the dense layer but does not initialize weights and biases.

func Dense(inputSize, numNeurons int, activation ActivationFunction) *DenseLayer {
	layer := NewLayer(inputSize, numNeurons, activation)

	return &DenseLayer{
		Layer: layer,
	}
}
