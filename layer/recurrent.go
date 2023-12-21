package layer

// Recurrent layers are designed to process sequences of data by maintaining a 'memory'
// of previous inputs through internal states.
//
// Fields:
//   - Layer: A pointer to the base Layer struct (includes fields like weights, biases, and activation function).
//   - TimeSteps: The number of time steps or sequence length the layer will unroll for during processing.
//
// Usage:
//   - Recurrent layers are good for processing time-series data, natural language processing, and other sequence-related tasks.
//
// Note:
//   - Unlike dense layers, recurrent layers share weights across time steps.

type RecurrentLayer struct {
	*Layer
	TimeSteps int
}

// Rec initializes a new instance of a recurrent layer with specified parameters.
//
// Arguments:
//   - inputSize: Number of input units to the layer (depends on the size of the previous layer or the feature space).
//   - numNeurons: Number of neurons (or units) in the recurrent layer.
//   - timeSteps: The number of time steps the layer should process.
//   - activation: Activation function to be used in the layer.
//
// Returns:
//   - A pointer to an initialized RecurrentLayer instance.
//
// Note:
//   - The timeSteps parameter determines how many previous steps in the sequence the layer retains information from.
//   - The activation function is applied to the output at each time step.
//   - Sets up the initial structure of the recurrent layer but does not initialize weights and biases.

func Rec(inputSize, numNeurons, timeSteps int, activation ActivationFunction) *RecurrentLayer {
	layer := NewLayer(inputSize, numNeurons, activation)

	return &RecurrentLayer{
		Layer:     layer,
		TimeSteps: timeSteps,
	}
}
