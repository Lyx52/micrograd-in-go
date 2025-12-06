package nn

type NormalizeLayer struct {
	context *NeuralContext
}

func NewNormalizeLayer(context *NeuralContext) *NormalizeLayer {
	return &NormalizeLayer{context: context}
}

func (layer *NormalizeLayer) Zerograd() {
}

func (layer *NormalizeLayer) UpdateParameters(updateCallback UpdateTensorFunction) {
}

func (layer *NormalizeLayer) Execute(tensor *Tensor) (*Tensor, error) {
	res, err := TensorDiv(tensor, tensor.MaxValue())

	return res, err
}
