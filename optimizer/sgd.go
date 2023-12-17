package optimizer

type SGDOptimizer struct {
	LearningRate float64
}

func SGD(learningRate float64) *SGDOptimizer {
	return &SGDOptimizer{
		LearningRate: learningRate,
	}
}

func (sgd *SGDOptimizer) Update(weights, gradients [][]float64, biases, biasGradients []float64) {
	// Update weights
	for i := range weights {
		for j := range weights[i] {
			weights[i][j] -= sgd.LearningRate * gradients[i][j]
		}
	}

	// Update biases
	for i := range biases {
		biases[i] -= sgd.LearningRate * biasGradients[i]
	}
}
