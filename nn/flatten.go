package nn

type FlattenLayer struct {
	context *NeuralContext
}

func NewFlattenLayer(context *NeuralContext) *FlattenLayer {
	return &FlattenLayer{context: context}
}

func (layer *FlattenLayer) Zerograd() {
}

func (layer *FlattenLayer) UpdateParameters(updateCallback UpdateTensorFunction) {
}

func (layer *FlattenLayer) Execute(tensor *Tensor) (*Tensor, error) {
	return tensor.Flatten(), nil
}
