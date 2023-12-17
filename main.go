package main

import (
	"fmt"

	"github.com/aaronsatko/gonenet/activation"
	"github.com/aaronsatko/gonenet/layer"
	"github.com/aaronsatko/gonenet/loss"
)

func main() {
	input := []float64{0.1, 0.2, 0.3}

	// dense layer with 4 neurons and ReLU activation function
	denseLayer := layer.Dense(len(input), 4, activation.ReLU)

	// forward pass
	output := denseLayer.Forward(input)

	fmt.Println("Layer Output:", output)

	yTrue := []float64{2.0, 4.0, 6.0, 8.0}
	yPred := []float64{1.5, 3.5, 5.5, 7.5}

	mse := loss.MeanSquaredError(yTrue, yPred)
	fmt.Printf("Mean Squared Error: %f\n", mse)
}
