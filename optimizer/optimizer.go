package optimizer

type Optimizer interface {
	Update(weights, gradients [][]float64, biases, biasGradients []float64)
}
