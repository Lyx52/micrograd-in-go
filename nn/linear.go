package nn

import (
	"errors"
	"fmt"
)

type LinearNeuron struct {
	Weights    *Tensor
	Bias       *Tensor
	activation ActivationFunction
	context    *NeuralContext
}

func NewLinearNeuron(context *NeuralContext, inputs int, useBias bool, activation ActivationFunction) *LinearNeuron {
	neuron := &LinearNeuron{
		activation: activation,
		context:    context,
		Weights:    NewTensorRand(inputs, context.Random),
	}

	if useBias {
		neuron.Bias = NewTensorRand(1, context.Random)
	}

	return neuron
}

func (neuron *LinearNeuron) UsesBias() bool {
	return neuron.Bias != nil
}

func (neuron *LinearNeuron) Zerograd() {
	neuron.Weights.Zerograd()

	if neuron.UsesBias() {
		neuron.Bias.Zerograd()
	}
}

func (neuron *LinearNeuron) UpdateParameters(updateCallback UpdateTensorFunction) {
	updateCallback(neuron.context, neuron.Weights)

	if neuron.UsesBias() {
		updateCallback(neuron.context, neuron.Bias)
	}
}

func (neuron *LinearNeuron) Execute(tensor *Tensor) (*Tensor, error) {
	if !neuron.Weights.TensorEqual(tensor) {
		return nil, errors.New(fmt.Sprintf("layer tensors dont match %v (Input) != %v (Target)", len(tensor.Gradients), len(neuron.Weights.Backing.Backing)))
	}

	sum, err := neuron.Weights.Mul(tensor)

	if err != nil {
		return nil, err
	}

	sum = sum.Sum()
	if neuron.UsesBias() {
		sum, err = sum.Add(neuron.Bias)
	}

	if err != nil {
		return nil, err
	}

	return neuron.activation(sum)
}

type LinearLayer struct {
	Neurons []*LinearNeuron
	context *NeuralContext
}

func NewLinearLayer(context *NeuralContext, inputs int, outputs int, useBias bool, activation ActivationFunction) *LinearLayer {
	layer := &LinearLayer{
		Neurons: make([]*LinearNeuron, outputs),
		context: context,
	}

	for i := 0; i < outputs; i++ {
		layer.Neurons[i] = NewLinearNeuron(context, inputs, useBias, activation)
	}

	return layer
}

func (layer *LinearLayer) Zerograd() {
	for i := range layer.Neurons {
		layer.Neurons[i].Zerograd()
	}
}

func (layer *LinearLayer) UpdateParameters(updateCallback UpdateTensorFunction) {
	for i := range layer.Neurons {
		layer.Neurons[i].UpdateParameters(updateCallback)
	}
}

func (layer *LinearLayer) Execute(tensor *Tensor) (*Tensor, error) {
	results := make([]*Tensor, len(layer.Neurons))
	var err error
	var result *Tensor

	for i := range layer.Neurons {
		result, err = layer.Neurons[i].Execute(tensor)
		if err != nil {
			return nil, err
		}

		results[i] = result
	}

	return NewFromTensors(results), nil
}
