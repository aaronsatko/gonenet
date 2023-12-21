package main

import (
	"fmt"

	"github.com/aaronsatko/gonenet/activation"
	"github.com/aaronsatko/gonenet/layer"
	"github.com/aaronsatko/gonenet/loss"
	"github.com/aaronsatko/gonenet/optimizer"
	"github.com/aaronsatko/gonenet/datahandling"
)

func main() {
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
        for i, batch := range batches {
            inputData := batch
            trueOutput := labelBatches[i]

            // Forward pass
            inputLayerOutput := inputLayer.Forward(inputData)
            hiddenLayerOutput := hiddenLayer.Forward(inputLayerOutput)
            output := outputLayer.Forward(hiddenLayerOutput)

            // Calculate loss
            lossValue := loss.CrossEntropy(trueOutput, output)

            // Backward pass (compute gradients)
            outputLayerGradient := loss.Gradient(trueOutput, output)
            hiddenLayerGradient := outputLayer.Backward(outputLayerGradient)
            inputLayerGradient := hiddenLayer.Backward(hiddenLayerGradient)

            // Update weights and biases
            adam.Update(inputLayer.Weights, inputLayerGradient, inputLayer.Biases, [bias gradients])
            adam.Update(hiddenLayer.Weights, hiddenLayerGradient, hiddenLayer.Biases, [bias gradients])
            adam.Update(outputLayer.Weights, outputLayerGradient, outputLayer.Biases, [bias gradients])

            // [Rest of your training loop]
        }
    }

    // [Model evaluation]
}

	// need to eval model on test data


