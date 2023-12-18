package datahandling

import (
	"encoding/csv"
	"os"
	"strconv"
)

// LoadData loads data from a CSV file
func LoadData(filePath string) ([]float64, []float64) {
	file, err := os.Open(filePath)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	var data []float64
	var labels []float64

	for {
		record, err := reader.Read()
		if err != nil {
			break
		}

		// Assuming the last column is the label
		for i, value := range record {
			floatValue, _ := strconv.ParseFloat(value, 64)
			if i == len(record)-1 {
				labels = append(labels, floatValue)
			} else {
				data = append(data, floatValue)
			}
		}
	}

	return data, labels
}
