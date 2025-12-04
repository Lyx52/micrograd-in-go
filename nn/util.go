package nn

import (
	"math/rand"
)

func Fill[T any](arr []T, value T) []T {
	for i := 0; i < len(arr); i++ {
		arr[i] = value
	}

	return arr
}

func MapFloat64Slice(arr []float64, mapCallback func(float64, int) float64) []float64 {
	result := make([]float64, len(arr))
	for i := 0; i < len(arr); i++ {
		result[i] = mapCallback(arr[i], i)
	}

	return result
}

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

func sampleIndices(arr []*Tensor, count int) []int {
	indices := make([]int, len(arr))
	for i := range arr {
		indices[i] = i
	}

	if len(indices) < count {
		return indices
	}

	rand.Shuffle(len(indices), func(i, j int) {
		indices[i], indices[j] = indices[j], indices[i]
	})

	return indices[:count]
}

func SampleDataset(ys []*Tensor, xs []*Tensor, batchSize int) ([]*Tensor, []*Tensor) {
	indices := sampleIndices(xs, batchSize)
	samplesX, samplesY := make([]*Tensor, len(indices)), make([]*Tensor, len(indices))
	for i := 0; i < len(indices); i++ {
		samplesX[i] = xs[indices[i]]
		samplesY[i] = ys[indices[i]]
	}

	return samplesX, samplesY
}
