package activation

func SingleReLU(x float64) float64 {
	if x < 0 {
		return 0
	}
	return x
}

func ReLU(input []float64) []float64 {
	output := make([]float64, len(input))
	for i, val := range input {
		output[i] = SingleReLU(val)
	}
	return output
}
