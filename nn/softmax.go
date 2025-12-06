package nn

type SoftmaxLayer struct {
	context *NeuralContext
}

func NewSoftmaxLayer(context *NeuralContext) *SoftmaxLayer {
	return &SoftmaxLayer{context: context}
}

func (layer *SoftmaxLayer) Zerograd() {
}

func (layer *SoftmaxLayer) UpdateParameters(updateCallback UpdateTensorFunction) {
}

func (layer *SoftmaxLayer) Execute(tensor *Tensor) (*Tensor, error) {
	return tensor.Softmax()
}
