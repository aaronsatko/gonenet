package activation

import (
	"math"
)

// vanishing gradient problem?

func Sigmoid(X float64) float64 {
	return 1 / (1 + math.Exp(-X))
}
