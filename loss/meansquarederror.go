package loss

import "math"

// MeanSquaredError measures the average of the squares of the errors.
//
// Usage:
//   - Primarily used in regression models to quantify difference between predicted values and actual values.
//   - It can also be used in neural networks where the output layer consists of neurons representing continuous values.
//
// Arguments:
//   - yTrue: A slice of float64 representing the true values.
//   - yPred: A slice of float64 representing the predicted values (outputs).
//
// Returns:
//   - A float64 value representing the MSE across all elements of the input slices.
//
// Panics:
//   - If yTrue and yPred have different lengths, the function will panic.
//     Both slices must have the same number of elements to compute the MSE correctly.
//
// Note:
//   - MSE is sensitive to outliers. Large errors are penalized more than smaller errors, as the error is squared.

func MeanSquaredError(yTrue, yPred []float64) float64 {
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
