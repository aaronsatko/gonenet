package datahandling

import (
	"fmt"
	"math/rand"
	"time"
)

func SplitData(data, labels []float64, trainRatio float64) ([]float64, []float64, []float64, []float64, error) {
	if len(data) != len(labels) {
		return nil, nil, nil, nil, fmt.Errorf("data and labels must be of the same length")
	}

	// Shuffle the data and labels
	rand.Seed(time.Now().UnixNano())
	rand.Shuffle(len(data), func(i, j int) {
		data[i], data[j] = data[j], data[i]
		labels[i], labels[j] = labels[j], labels[i]
	})

	// Determine the split index
	splitIndex := int(float64(len(data)) * trainRatio)

	// Split the data and labels into training and test sets
	trainData := data[:splitIndex]
	testData := data[splitIndex:]
	trainLabels := labels[:splitIndex]
	testLabels := labels[splitIndex:]

	return trainData, testData, trainLabels, testLabels, nil
}
