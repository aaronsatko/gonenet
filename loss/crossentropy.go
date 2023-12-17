package loss

import "math"

// classification tasks
// dif between predicted class probibilities and true class values

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
