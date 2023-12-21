package optimizer

// Stochastic Gradient Descent optimization algorithm.
// Fits linear classifiers and neural network weights.
//
// Fields:
//   - LearningRate: Step size used for each iteration of optimization.
//
// Usage:
//   - Used in training often when data is sparse.
//   - Suitable for both online and batch training methods.
//
// Note:
//   - SGD updates parameters in the opposite direction of the gradient of the objective function.
//   - It is a fundamental optimization technique and a building block for many other complex optimizers.

type SGDOptimizer struct {
	LearningRate float64
}

// Initializes a new instance of SGDOptimizer with the specified learning rate.
//
// Arguments:
//   - learningRate: A float64 value representing the step size for each iteration of optimization.
//
// Returns:
//   - A pointer to an initialized SGDOptimizer instance.

func SGD(learningRate float64) *SGDOptimizer {
	return &SGDOptimizer{
		LearningRate: learningRate,
	}
}

// Updates the weights and biases of the neural network based on the gradients.
//
// Usage:
//   - This method should be called after computing gradients for each batch during the training process.
//
// Arguments:
//   - weights: A slice of slices containing the current weights of the neural network.
//   - gradients: A slice of slices containing the gradients of the loss function with respect to each weight.
//   - biases: A slice containing the current biases of the neural network.
//   - biasGradients: A slice containing the gradients of the loss function with respect to each bias.
//
// Note:
//   - This method modifies the weights and biases in place, representing a single optimization step.

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
