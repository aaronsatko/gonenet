package layer

// ConvLayer represents a convolutional layer in a neural network.
// Convolutional layers are primarily used in processing data with a grid-like topology (e.g., images).
//
// Fields:
//   - Layer: A pointer to the base Layer struct (includes fields like weights, biases, and activation function).
//   - KernelSize: Size of the convolutional kernel (filter).
//   - Stride: Step size with which the kernel moves across the input data.
//
// Usage:
//   - Convolutional layers are essential in Convolutional Neural Networks (CNNs) for tasks like image classification, object detection, etc.
//   - They are effective in extracting features from local input patches and maintaining spatial relationships between pixels.
//
// Note:
//   - The convolution operation involves sliding the kernel over the input and computing dot products.
//   - KernelSize and Stride determine how the kernel moves across the input and how the output dimensions are calculated.

type ConvLayer struct {
	*Layer
	KernelSize int
	Stride     int
}

// Conv creates a new instance of a convolutional layer with specified parameters.
//
// Arguments:
//   - inputSize: Number of input units to the layer (e.g., number of pixels in an image).
//   - numNeurons: Number of neurons (or filters) in the convolutional layer.
//   - kernelSize: Size of the convolutional kernel (filter).
//   - stride: Step size for moving the kernel over the input data.
//   - activation: Activation function to be used in the layer.
//
// Returns:
//   - A pointer to an initialized ConvLayer instance.
//
// Note:
//   - The activation function is applied to the output of the convolution operation.
//   - This function sets up the initial structure of the convolutional layer but does not initialize weights and biases.

func Conv(inputSize, numNeurons, kernelSize, stride int, activation ActivationFunction) *ConvLayer {
	layer := NewLayer(inputSize, numNeurons, activation)

	return &ConvLayer{
		Layer:      layer,
		KernelSize: kernelSize,
		Stride:     stride,
	}
}
