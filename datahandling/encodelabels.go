package datahandling

func EncodeLabels(labels []string) ([]int, error) {
	if len(labels) == 0 {
		return nil, nil
	}

	labelEncodings := make(map[string]int)
	uniqueLabelIndex := 0

	encodedLabels := make([]int, len(labels))

	for i, label := range labels {
		if _, exists := labelEncodings[label]; !exists {
			labelEncodings[label] = uniqueLabelIndex
			uniqueLabelIndex++
		}
		encodedLabels[i] = labelEncodings[label]
	}

	return encodedLabels, nil
}
