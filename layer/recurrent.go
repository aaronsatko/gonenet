package layer

type RecurrentLayer struct {
	*Layer
	TimeSteps int
}

func Rec(inputSize, numNeurons, timeSteps int, activation ActivationFunction) *RecurrentLayer {
	layer := NewLayer(inputSize, numNeurons, activation)

	return &RecurrentLayer{
		Layer:     layer,
		TimeSteps: timeSteps,
	}
}
