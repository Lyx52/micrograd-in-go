package nn

type Module struct {
	Layers  []ICallable
	context *NeuralContext
}

func NewModule(context *NeuralContext, layers ...ICallable) *Module {
	return &Module{
		Layers:  layers,
		context: context,
	}
}

func (module *Module) Zerograd() {
	for i := range module.Layers {
		module.Layers[i].Zerograd()
	}
}

func (module *Module) UpdateParameters(updateCallback UpdateTensorFunction) {
	for i := range module.Layers {
		module.Layers[i].UpdateParameters(updateCallback)
	}
}

func (module *Module) Execute(tensor *Tensor) (*Tensor, error) {
	result := tensor
	var err error

	for i := range module.Layers {
		result, err = module.Layers[i].Execute(result)
		if err != nil {
			return nil, err
		}
	}

	return result, nil
}
