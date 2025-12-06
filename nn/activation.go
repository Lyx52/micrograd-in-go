package nn

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
