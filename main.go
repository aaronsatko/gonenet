package main

import (
	"fmt"

	"github.com/aaronsatko/gonenet/activation"
	"github.com/aaronsatko/gonenet/layer"
)

func main() {
	input := []float64{0.1, 0.2, 0.3}

	// dense layer with 4 neurons and ReLU activation function
	denseLayer := layer.Dense(len(input), 4, activation.ReLU)

	// forward pass
	output := denseLayer.Forward(input)

	fmt.Println("Layer Output:", output)
}
