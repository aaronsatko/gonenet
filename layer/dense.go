package layer

type DenseLayer struct {
	*Layer
}

func Dense(inputSize, numNeurons int, activation ActivationFunction) *DenseLayer {
	layer := NewLayer(inputSize, numNeurons, activation)

	return &DenseLayer{
		Layer: layer,
	}
}
