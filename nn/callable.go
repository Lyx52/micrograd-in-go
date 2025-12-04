package nn

type ICallable interface {
	Execute(tensor *Tensor) (*Tensor, error)
	Zerograd()
	UpdateParameters(updateCallback UpdateTensorFunction)
}

type UpdateTensorFunction func(*NeuralContext, *Tensor)

type ActivationFunction func(tensor *Tensor) (*Tensor, error)

func NoneActivation(tensor *Tensor) (*Tensor, error) {
	return tensor, nil
}

func TanhActivation(tensor *Tensor) (*Tensor, error) {
	return tensor.Tanh(), nil
}

func ReluActivation(tensor *Tensor) (*Tensor, error) {
	return tensor.Relu(), nil
}
