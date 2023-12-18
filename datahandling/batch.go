package datahandling

import "fmt"

func Batch(data []float64, batchSize int) ([][]float64, error) {
	if batchSize <= 0 {
		return nil, fmt.Errorf("batch size must be positive")
	}

	// Calc num batches
	numBatches := (len(data) + batchSize - 1) / batchSize

	// slice to hold batches
	batches := make([][]float64, 0, numBatches)

	for start := 0; start < len(data); start += batchSize {
		end := start + batchSize
		if end > len(data) {
			end = len(data)
		}

		batch := make([]float64, end-start)
		copy(batch, data[start:end])
		batches = append(batches, batch)
	}

	return batches, nil
}
