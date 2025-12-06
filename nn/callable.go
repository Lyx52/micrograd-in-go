package nn

type ICallable interface {
	Execute(tensor *Tensor) (*Tensor, error)
	Zerograd()
	UpdateParameters(updateCallback UpdateTensorFunction)
}

type UpdateTensorFunction func(*NeuralContext, *Tensor)
