package backpropagation

import (
	"github.com/aaronsatko/gonenet/layer"
	"github.com/aaronsatko/gonenet/loss"
)

type Backpropagation struct {
	Layers       []layer.Layer
	LossFunction loss.LossFunction
}

func (bp *Backpropagation) Forward(input []float64) []float64 {
	// Implement the forward propagation
}

func (bp *Backpropagation) ComputeLoss(target []float64) float64 {
	// Implement loss calculation
}

func (bp *Backpropagation) Backward(target []float64) {
	// Implement the backward propagation
}

func (bp *Backpropagation) UpdateWeights(learningRate float64) {
	// Implement the weights update
}
