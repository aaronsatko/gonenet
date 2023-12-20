# GoNeNet: A Neural Network Framework in Go

GoNeNet is a lightweight, flexible neural network framework written in Go, designed for both educational purposes and practical machine learning projects. With its modular design, it offers a simple yet powerful set of tools to build, train, and evaluate neural networks.

## Features

- **Layer Support**: Includes various types of layers like dense, convolutional, and recurrent.
- **Activation Functions**: ReLU and Sigmoid activation functions are implemented, with room for easy extension.
- **Loss Functions**: Supports Cross Entropy and Mean Squared Error for different types of neural network tasks.
- **Optimizers**: Features popular optimization algorithms Adam and SGD.
- **Data Handling**: Utility functions for data loading, preprocessing, batching, and splitting.
- **Metrics**: (Not yet Implemented)

## Installation

Ensure you have Go installed on your system. Then, you can clone this repository:

```bash
git clone https://github.com/aaronsatko/gonenet
```

Navigate to the cloned directory:

```bash
cd gonenet
```

## Usage

Here is a basic example of using GoNeNet to create a simple neural network:

```go
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


```

## Documentation

For detailed documentation on each component of the framework, refer to the comments in the respective Go files. Each package and function is documented to explain its purpose and usage.

