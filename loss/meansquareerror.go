package loss

import "math"

func MeanSquareError(yTrue, yPred []float64) float64 {
	if len(yTrue) != len(yPred) {
		panic("Input arrays must have the same length")
	}

	sumSqError := 0.0

	for i := 0; i < len(yTrue); i++ {
		sqError := math.Pow(yTrue[i]-yPred[i], 2)
		sumSqError += sqError
	}

	mse := sumSqError / float64(len(yTrue))
	return mse
}
