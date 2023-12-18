package model

import (
	"errors"

	"github.com/aaronsatko/gonenet/layer"
	"github.com/aaronsatko/gonenet/optimizer"
)

type ModelState int

const (
	Uncompiled ModelState = iota
	Compiled
	Trained
)

type Model struct {
	Layers    []layer.Layer
	Loss      func([]float64, []float64) float64
	Optimizer optimizer.Optimizer
	State     ModelState
}

func NewModel() *Model {
	return &Model{State: Uncompiled}
}

func (m *Model) AddLayer(l layer.Layer) error {
	if len(m.Layers) > 0 {
		lastLayer := m.Layers[len(m.Layers)-1]
		// Assuming each layer struct has an OutputSize method
		if l.InputSize() != lastLayer.OutputSize() {
			return errors.New("input size of the new layer does not match the output size of the last layer")
		}
	}
	m.Layers = append(m.Layers, l)
	return nil
}

func (m *Model) Compile(lossFunc func([]float64, []float64) float64, opt optimizer.Optimizer) {
	m.Loss = lossFunc
	m.Optimizer = opt
	m.State = Compiled
}

func (m *Model) Fit(xTrain [][]float64, yTrain [][]float64, epochs int) error {
	if m.State != Compiled {
		return errors.New("model is not compiled")
	}
	// Training logic here...
	// Remember to update model state to Trained after successful training
	m.State = Trained
	return nil
}

func (m *Model) Predict(x []float64) ([]float64, error) {
	if m.State != Trained {
		return nil, errors.New("model is not trained")
	}
	// Prediction logic here...
	return nil, nil
}

// Placeholder for future method implementations
func (m *Model) Evaluate(xTest [][]float64, yTest [][]float64) float64 {
	// Evaluation logic here...
	return 0.0
}

func (m *Model) Save(filePath string) error {
	// Save model logic here...
	return nil
}

func (m *Model) Load(filePath string) error {
	// Load model logic here...
	return nil
}
