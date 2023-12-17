package layer

type ConvLayer struct {
	*Layer
	KernelSize int
	Stride     int
}

func Conv(inputSize, numNeurons, kernelSize, stride int, activation ActivationFunction) *ConvLayer {
	layer := NewLayer(inputSize, numNeurons, activation)

	return &ConvLayer{
		Layer:      layer,
		KernelSize: kernelSize,
		Stride:     stride,
	}
}
