package main

import (
	"fmt"

	"github.com/aaronsatko/gonenet/activation"
	"github.com/aaronsatko/gonenet/layer"
	"github.com/aaronsatko/gonenet/loss"
	"github.com/aaronsatko/gonenet/optimizer"
	"github.com/aaronsatko/gonenet/datahandling"
)

func man() {
	// Load and preprocess data
	data, labels := datahandling.LoadData("path/to/your/data.csv")
	data = datahandling.Preprocess(data)

	// Assume labels are in a format that needs encoding
	encodedLabels, err := datahandling.EncodeLabels(labels)
	if err != nil {
		fmt.Println("Error encoding labels:", err)
		return
	}

	// Split data into training and test sets
	trainData, testData, trainLabels, testLabels, err := datahandling.SplitData(data, encodedLabels, 0.8)
	if err != nil {
		fmt.Println("Error splitting data:", err)
		return
	}

	// Define nn architecture
	inputLayer := layer.Dense(inputSize, numNeurons, activation.ReLU)
	hiddenLayer := layer.Dense(numNeurons, hiddenLayerNeurons, activation.Sigmoid)
	outputLayer := layer.Dense(hiddenLayerNeurons, outputNeurons, activation.Sigmoid)

	// Initialize optimizer
	adam := optimizer.Adam(learningRate, beta1, beta2, epsilon, weightShape, biasShape)

	// Training loop
	for epoch := 0; epoch < numEpochs; epoch++ {
		// Create batches
		batches, _ := datahandling.Batch(trainData, batchSize)
		labelBatches, _ := datahandling.Batch(trainLabels, batchSize)

		for i, batch := range batches {
			inputData := batch
			trueOutput := labelBatches[i]

			// Forward pass through each layer
			inputLayerOutput := inputLayer.Forward(inputData)
			hiddenLayerOutput := hiddenLayer.Forward(inputLayerOutput)
			output := outputLayer.Forward(hiddenLayerOutput)

			// Calculate loss
			lossValue := loss.CrossEntropy(trueOutput, output)

			// implement backprop

			adam.Update(...) // Update weights and biases

			// need to add functions for loss, accuracy
		}
	}

	// need to eval model on test data
}

