package optimizer

import "math"

// AdamOptimizer represents the Adam optimization algorithm, which is an extension to stochastic gradient descent.
// Computes adaptive learning rates for each parameter.
//
// Fields:
//   - LearningRate: Step size used for each iteration of optimization.
//   - Beta1, Beta2: Exponential decay rates for the moment estimates (typically close to 1).
//   - Epsilon: A small constant for numerical stability.
//   - M, V: First (mean) and second (variance) moment vectors for weights.
//   - MBias, VBias: First and second moment vectors for biases.
//   - T: Time step (or iteration number), used for bias correction.
//
// Usage:
//   - This optimizer should be used in conjunction with backpropagation to train neural networks.
//   - Particularly effective for problems with noisy or sparse gradients.
//
// Note:
//   - Adam combines the advantages of two other extensions of stochastic gradient descent: AdaGrad and RMSProp.

type Optimizer interface {
	Update(weights, gradients [][]float64, biases, biasGradients []float64)
}

type AdamOptimizer struct {
	LearningRate float64
	Beta1        float64
	Beta2        float64
	Epsilon      float64
	M            [][]float64 // mean for weights
	V            [][]float64 // variance for weights
	MBias        []float64   // mean for biases
	VBias        []float64   // variance for biases
	T            int         // Time step
}

func Adam(learningRate, beta1, beta2, epsilon float64, weightShape [][]float64, biasShape []int) *AdamOptimizer {
	// Initialize the optimizer for weights
	m := make([][]float64, len(weightShape))
	v := make([][]float64, len(weightShape))
	for i := range weightShape {
		m[i] = make([]float64, len(weightShape[i]))
		v[i] = make([]float64, len(weightShape[i]))
	}

	// Initialize the optimizer for biases
	mBias := make([]float64, len(biasShape))
	vBias := make([]float64, len(biasShape))

	return &AdamOptimizer{
		LearningRate: learningRate,
		Beta1:        beta1,
		Beta2:        beta2,
		Epsilon:      epsilon,
		M:            m,
		V:            v,
		MBias:        mBias,
		VBias:        vBias,
		T:            0,
	}
}

func (adam *AdamOptimizer) Update(weights, gradients [][]float64, biases, biasGradients []float64) {
	// Increment the time step
	adam.T++

	// Update weights
	for i := range weights {
		for j := range weights[i] {
			adam.M[i][j] = adam.Beta1*adam.M[i][j] + (1.0-adam.Beta1)*gradients[i][j]
			adam.V[i][j] = adam.Beta2*adam.V[i][j] + (1.0-adam.Beta2)*gradients[i][j]*gradients[i][j]

			mHat := adam.M[i][j] / (1.0 - math.Pow(adam.Beta1, float64(adam.T)))
			vHat := adam.V[i][j] / (1.0 - math.Pow(adam.Beta2, float64(adam.T)))

			weights[i][j] -= adam.LearningRate * mHat / (math.Sqrt(vHat) + adam.Epsilon)
		}
	}

	// Update biases
	for i := range biases {
		adam.MBias[i] = adam.Beta1*adam.MBias[i] + (1.0-adam.Beta1)*biasGradients[i]
		adam.VBias[i] = adam.Beta2*adam.VBias[i] + (1.0-adam.Beta2)*biasGradients[i]*biasGradients[i]

		mHat := adam.MBias[i] / (1.0 - math.Pow(adam.Beta1, float64(adam.T)))
		vHat := adam.VBias[i] / (1.0 - math.Pow(adam.Beta2, float64(adam.T)))

		biases[i] -= adam.LearningRate * mHat / (math.Sqrt(vHat) + adam.Epsilon)
	}
}
