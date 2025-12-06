package nn

type LossFunction = func(module ICallable, ys []*Tensor, xs []*Tensor, batchSize int) (*Tensor, error)

func CrossEntropyLoss(module ICallable, ys []*Tensor, xs []*Tensor, batchSize int) (*Tensor, error) {
	samplesX, samplesY := SampleDataset(ys, xs, batchSize)
	losses := make([]*Tensor, len(samplesX))
	for i := 0; i < len(samplesX); i++ {
		result, err := module.Execute(samplesX[i])
		if err != nil {
			return nil, err
		}

		losses[i], err = result.Mse(samplesY[i])
		if err != nil {
			return nil, err
		}
	}
	result, err := TensorDiv(NewFromTensors(losses).Sum(), float64(len(losses)))
	if err != nil {
		return nil, err
	}

	return result, nil
}

func CategoricalCrossEntropyLoss(module ICallable, ys []*Tensor, xs []*Tensor, batchSize int) (*Tensor, error) {
	samplesX, samplesY := SampleDataset(ys, xs, batchSize)
	losses := make([]*Tensor, len(samplesX))

	for i := 0; i < len(samplesX); i++ {
		result, err := module.Execute(samplesX[i])
		if err != nil {
			return nil, err
		}

		losses[i], err = result.CategoricalCrossEntropy(samplesY[i])
		if err != nil {
			return nil, err
		}
	}
	result, err := TensorDiv(NewFromTensors(losses).Sum(), float64(len(losses)))
	if err != nil {
		return nil, err
	}

	return result, nil
}
