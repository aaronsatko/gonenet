package loss

import "math"

// CrossEntropy calculates the cross entropy loss to measure the difference between two probability distributions - the true label distribution and the predicted distribution.
//
// Usage:
//   - It is suitable for binary and multi-class classification tasks where the output is normalized to represent probabilities.
//
// Arguments:
//   - yTrue: A slice of float64 representing the true class values (labels).
//            For binary classification, these are usually 0 (negative class) or 1 (positive class).
//   - yPred: A slice of float64 representing the predicted class probabilities.
//            Each element should be a probability value between 0 and 1.
//
// Returns:
//   - A float64 value representing the average cross entropy loss across all elements of the input slices.
//
// Panics:
//   - If yTrue and yPred have different lengths, the function will panic.
//     Both slices must have the same number of elements.
//
// Note:
//   - The function applies a negative log likelihood loss to each pair of true and predicted values,
//     summing them up and then dividing by the number of elements to get the average loss.
//   - It's important for yPred values to be within (0,1), as the log of 0 or values outside this range will result in errors.

func CrossEntropy(yTrue, yPred []float64) float64 {
	if len(yTrue) != len(yPred) {
		panic("Error: Input arrays must have same length")
	}

	loss := 0.0
	for i := 0; i < len(yTrue); i++ {
		loss += -yTrue[i]*math.Log(yPred[i]) - (1-yTrue[i])*math.Log(1-yPred[i])
	}

	loss /= float64(len(yTrue))

	return loss
}
