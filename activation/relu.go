package activation

func ReLU(x float64) float64 {
	if x < 0 {
		return 0
	}
	return x
}

func ArrRelu(input []float64) []float64 {
	output := make([]float64, len(input))
	for i, val := range input {
		output[i] = ReLU(val)
	}
	return output
}
