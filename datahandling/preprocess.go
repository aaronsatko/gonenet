package datahandling

func Preprocess(data []float64) []float64 {
	if len(data) == 0 {
		return data
	}

	// Find min and max vals
	minVal, maxVal := data[0], data[0]
	for _, value := range data {
		if value < minVal {
			minVal = value
		}
		if value > maxVal {
			maxVal = value
		}
	}

	// avoid division by zero
	if maxVal == minVal {
		return make([]float64, len(data))
	}

	// Normalize
	normalizedData := make([]float64, len(data))
	for i, value := range data {
		normalizedData[i] = (value - minVal) / (maxVal - minVal)
	}

	return normalizedData
}
